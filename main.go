package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/spf13/viper"
	"github.com/xuri/excelize/v2"
)

// Config represents the application configuration
type Config struct {
	Files struct {
		Projects string
		Turtles  string
		Output   string
	}
	Analysis struct {
		DistanceThresholds []float64
	}
	Performance struct {
		UseSpatialIndex bool
		UseParallel     bool
		BatchSize       int
	}
}

// Project represents a dredging project
type Project struct {
	StartDate time.Time
	EndDate   time.Time
	Latitude  float64
	Longitude float64
}

// Turtle represents a sea turtle stranding event
type Turtle struct {
	ID         string
	ReportDate time.Time
	Latitude   float64
	Longitude  float64
	// Original data for output
	OriginalData map[string]string
}

// SpatialGrid implements a simple spatial indexing structure
type SpatialGrid struct {
	cellSize float64
	grid     map[string][]int
	turtles  []Turtle
}

func newSpatialGrid(turtles []Turtle, cellSize float64) *SpatialGrid {
	sg := &SpatialGrid{
		cellSize: cellSize,
		grid:     make(map[string][]int),
		turtles:  turtles,
	}

	// Index turtles in the grid
	for i, turtle := range turtles {
		cellKey := sg.getCellKey(turtle.Latitude, turtle.Longitude)
		sg.grid[cellKey] = append(sg.grid[cellKey], i)
	}

	return sg
}

func (sg *SpatialGrid) getCellKey(lat, lng float64) string {
	cellLat := int(lat / sg.cellSize)
	cellLng := int(lng / sg.cellSize)
	return fmt.Sprintf("%d:%d", cellLat, cellLng)
}

func (sg *SpatialGrid) findNearbyTurtles(lat, lng, threshold float64, startDate, endDate time.Time) []int {
	// Calculate the number of cells to check based on threshold
	cellsToCheck := int(threshold/sg.cellSize) + 1
	centralCellLat := int(lat / sg.cellSize)
	centralCellLng := int(lng / sg.cellSize)

	var nearbyTurtles []int

	// Check cells in a square around the central cell
	for dLat := -cellsToCheck; dLat <= cellsToCheck; dLat++ {
		for dLng := -cellsToCheck; dLng <= cellsToCheck; dLng++ {
			cellKey := fmt.Sprintf("%d:%d", centralCellLat+dLat, centralCellLng+dLng)

			// Get turtle indices in this cell
			indices, exists := sg.grid[cellKey]
			if !exists {
				continue
			}

			// Check each turtle in this cell
			for _, idx := range indices {
				turtle := sg.turtles[idx]

				// Check date range
				if turtle.ReportDate.Before(startDate) || turtle.ReportDate.After(endDate) {
					continue
				}

				// Calculate distance
				distance := haversine(lat, lng, turtle.Latitude, turtle.Longitude)

				if distance <= threshold {
					nearbyTurtles = append(nearbyTurtles, idx)
				}
			}
		}
	}

	return nearbyTurtles
}

// haversine calculates the distance in kilometers between two points on Earth
func haversine(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371.0 // Earth radius in kilometers

	lat1Rad := toRadians(lat1)
	lon1Rad := toRadians(lon1)
	lat2Rad := toRadians(lat2)
	lon2Rad := toRadians(lon2)

	dLat := lat2Rad - lat1Rad
	dLon := lon2Rad - lon1Rad

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
		math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}

func toRadians(degrees float64) float64 {
	return degrees * math.Pi / 180
}

func loadConfig(configPath string) Config {
	var config Config

	// Set default values
	viper.SetDefault("files.projects", "data/project_summary_export.xlsx")
	viper.SetDefault("files.turtles", "data/STSSN_report.xlsx")
	viper.SetDefault("files.output", "output/updated_STSSN_report.xlsx")
	viper.SetDefault("analysis.distance_thresholds", []float64{5, 10, 15})
	viper.SetDefault("performance.use_spatial_index", true)
	viper.SetDefault("performance.use_parallel", true)
	viper.SetDefault("performance.batch_size", 1000)

	// Read config file
	viper.SetConfigFile(configPath)
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// Create default config
			dir := filepath.Dir(configPath)
			if _, err := os.Stat(dir); os.IsNotExist(err) {
				os.MkdirAll(dir, 0755)
			}
			viper.WriteConfigAs(configPath)
			fmt.Printf("Created default config file at %s\n", configPath)
		} else {
			fmt.Printf("Error reading config: %s\n", err)
			fmt.Println("Using default configuration instead.")
		}
	}

	// Unmarshal config into struct
	if err := viper.Unmarshal(&config); err != nil {
		fmt.Printf("Error unmarshaling config: %s\n", err)
	}

	return config
}

// readExcelFile reads an Excel file and extracts projects or turtles
func readExcelFile(filePath string, isProject bool) (interface{}, []string, error) {
	fmt.Printf("Reading file: %s\n", filePath)

	file, err := excelize.OpenFile(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening Excel file: %w", err)
	}
	defer file.Close()

	// Get the first sheet
	sheetName := file.GetSheetList()[0]
	rows, err := file.GetRows(sheetName)
	if err != nil {
		return nil, nil, fmt.Errorf("error reading rows: %w", err)
	}

	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("file has insufficient data")
	}

	// Get header
	header := rows[0]

	// Standardize header names and create mapping
	headerMap := make(map[string]int)
	for i, colName := range header {
		colNameLower := strings.ToLower(colName)
		// Map common variations
		if isProject {
			switch {
			case strings.Contains(colNameLower, "start") || strings.Contains(colNameLower, "dqmstart"):
				headerMap["dqm_start_date"] = i
			case strings.Contains(colNameLower, "end") || strings.Contains(colNameLower, "dqmend"):
				headerMap["dqm_end_date"] = i
			case strings.Contains(colNameLower, "lat") && (strings.Contains(colNameLower, "dredg") || strings.Contains(colNameLower, "project")):
				headerMap["dredging_lat"] = i
			case strings.Contains(colNameLower, "lng") || strings.Contains(colNameLower, "long") && (strings.Contains(colNameLower, "dredg") || strings.Contains(colNameLower, "project")):
				headerMap["dredging_lng"] = i
			}
		} else { // Turtle
			switch {
			case strings.Contains(colNameLower, "stssn") || colNameLower == "id":
				headerMap["stssnid"] = i
			case strings.Contains(colNameLower, "report") || colNameLower == "date":
				headerMap["reportdate"] = i
			case colNameLower == "lat" || colNameLower == "latitude":
				headerMap["latitude"] = i
			case colNameLower == "lng" || colNameLower == "long" || colNameLower == "longitude":
				headerMap["longitude"] = i
			}
		}
	}

	// Validate required columns
	requiredCols := []string{}
	if isProject {
		requiredCols = []string{"dqm_start_date", "dqm_end_date", "dredging_lat", "dredging_lng"}
	} else {
		requiredCols = []string{"stssnid", "reportdate", "latitude", "longitude"}
	}

	for _, col := range requiredCols {
		if _, exists := headerMap[col]; !exists {
			return nil, nil, fmt.Errorf("required column '%s' not found", col)
		}
	}

	// Process rows
	if isProject {
		var projects []Project
		for i := 1; i < len(rows); i++ {
			row := rows[i]
			if len(row) <= max(headerMap["dqm_start_date"], headerMap["dqm_end_date"], headerMap["dredging_lat"], headerMap["dredging_lng"]) {
				continue // Skip rows with insufficient data
			}

			startDate, err := parseDate(row[headerMap["dqm_start_date"]])
			if err != nil {
				continue
			}

			endDate, err := parseDate(row[headerMap["dqm_end_date"]])
			if err != nil {
				continue
			}

			lat, err := strconv.ParseFloat(row[headerMap["dredging_lat"]], 64)
			if err != nil {
				continue
			}

			lng, err := strconv.ParseFloat(row[headerMap["dredging_lng"]], 64)
			if err != nil {
				continue
			}

			projects = append(projects, Project{
				StartDate: startDate,
				EndDate:   endDate,
				Latitude:  lat,
				Longitude: lng,
			})
		}
		return projects, header, nil
	} else {
		var turtles []Turtle
		for i := 1; i < len(rows); i++ {
			row := rows[i]
			if len(row) <= max(headerMap["stssnid"], headerMap["reportdate"], headerMap["latitude"], headerMap["longitude"]) {
				continue // Skip rows with insufficient data
			}

			id := row[headerMap["stssnid"]]

			reportDate, err := parseDate(row[headerMap["reportdate"]])
			if err != nil {
				continue
			}

			lat, err := strconv.ParseFloat(row[headerMap["latitude"]], 64)
			if err != nil {
				continue
			}

			lng, err := strconv.ParseFloat(row[headerMap["longitude"]], 64)
			if err != nil {
				continue
			}

			// Store original row data for output
			originalData := make(map[string]string)
			for j, col := range header {
				if j < len(row) {
					originalData[col] = row[j]
				} else {
					originalData[col] = ""
				}
			}

			turtles = append(turtles, Turtle{
				ID:           id,
				ReportDate:   reportDate,
				Latitude:     lat,
				Longitude:    lng,
				OriginalData: originalData,
			})
		}
		return turtles, header, nil
	}
}

// parseDate tries to parse a date string in multiple formats
func parseDate(dateStr string) (time.Time, error) {
	formats := []string{
		"2006-01-02",
		"1/2/2006",
		"01/02/2006",
		"1-2-2006",
		"01-02-2006",
		"2006/01/02",
		"2006-Jan-02",
		"Jan 2, 2006",
		"January 2, 2006",
		time.RFC3339,
	}

	for _, format := range formats {
		if t, err := time.Parse(format, dateStr); err == nil {
			return t, nil
		}
	}

	// Try Excel numeric date format
	if floatValue, err := strconv.ParseFloat(dateStr, 64); err == nil {
		// Excel dates start at January 1, 1900
		baseDate := time.Date(1899, 12, 30, 0, 0, 0, 0, time.UTC)
		days := int(floatValue)
		return baseDate.AddDate(0, 0, days), nil
	}

	return time.Time{}, fmt.Errorf("could not parse date: %s", dateStr)
}

func max(values ...int) int {
	if len(values) == 0 {
		return 0
	}

	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	return maxVal
}

// processChunk processes a chunk of projects against all turtles
func processChunk(chunk []Project, turtles []Turtle, threshold float64, results map[string]bool, wg *sync.WaitGroup, resMutex *sync.Mutex) {
	defer wg.Done()

	// Process each project in chunk
	for _, project := range chunk {
		// Add 10 days to end date
		endDatePlus10 := project.EndDate.AddDate(0, 0, 10)

		// Check each turtle
		for i, turtle := range turtles {
			// Skip if turtle report date is outside project timeframe
			if turtle.ReportDate.Before(project.StartDate) || turtle.ReportDate.After(endDatePlus10) {
				continue
			}

			// Calculate distance
			distance := haversine(project.Latitude, project.Longitude, turtle.Latitude, turtle.Longitude)

			// Check if within threshold
			if distance <= threshold {
				resMutex.Lock()
				results[turtle.ID] = true
				resMutex.Unlock()
			}
		}
	}
}

// processSpatialGrid uses spatial indexing for faster proximity checks
func processSpatialGrid(projects []Project, turtles []Turtle, threshold float64) map[string]bool {
	results := make(map[string]bool)

	// Create spatial grid with cell size based on threshold
	cellSize := threshold / 2 // A reasonable subdivision of the threshold distance
	grid := newSpatialGrid(turtles, cellSize)

	// Process each project
	for _, project := range projects {
		// Add 10 days to end date
		endDatePlus10 := project.EndDate.AddDate(0, 0, 10)

		// Find turtles near this project
		nearbyIndices := grid.findNearbyTurtles(project.Latitude, project.Longitude, threshold, project.StartDate, endDatePlus10)

		// Mark these turtles as within threshold
		for _, idx := range nearbyIndices {
			results[turtles[idx].ID] = true
		}
	}

	return results
}

// processParallel processes data in parallel using goroutines
func processParallel(projects []Project, turtles []Turtle, threshold float64) map[string]bool {
	numCPU := runtime.NumCPU()
	results := make(map[string]bool)

	// Create chunks of projects
	chunkSize := max(1, len(projects)/(numCPU*2))
	var chunks [][]Project
	for i := 0; i < len(projects); i += chunkSize {
		end := min(i+chunkSize, len(projects))
		chunks = append(chunks, projects[i:end])
	}

	// Process chunks in parallel
	var wg sync.WaitGroup
	var resMutex sync.Mutex

	for _, chunk := range chunks {
		wg.Add(1)
		go processChunk(chunk, turtles, threshold, results, &wg, &resMutex)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	return results
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// writeResults writes the analysis results to an Excel file
func writeResults(turtles []Turtle, thresholdResults map[float64]map[string]bool, header []string, outputPath string) error {
	// Create output directory if it doesn't exist
	dir := filepath.Dir(outputPath)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	// Determine if the output is Excel or CSV based on file extension
	ext := strings.ToLower(filepath.Ext(outputPath))

	if ext == ".xlsx" {
		return writeExcelOutput(turtles, thresholdResults, header, outputPath)
	} else {
		// Default to CSV if not Excel
		return writeCSVOutput(turtles, thresholdResults, header, outputPath)
	}
}

// writeExcelOutput writes results to an Excel file
func writeExcelOutput(turtles []Turtle, thresholdResults map[float64]map[string]bool, header []string, outputPath string) error {
	// Create a new Excel file
	f := excelize.NewFile()
	defer f.Close()

	// Create sheet
	sheetName := "Results"
	_, err := f.NewSheet(sheetName)
	if err != nil {
		return err
	}

	// Add threshold columns to header
	newHeader := append([]string{}, header...)
	for threshold := range thresholdResults {
		colName := fmt.Sprintf("near_project_%gkm", threshold)
		newHeader = append(newHeader, colName)
	}

	// Write header
	for i, col := range newHeader {
		cell, _ := excelize.CoordinatesToCellName(i+1, 1)
		f.SetCellValue(sheetName, cell, col)
	}

	// Write data
	for i, turtle := range turtles {
		row := i + 2 // Row 1 is header

		// Write original data
		for j, col := range header {
			cell, _ := excelize.CoordinatesToCellName(j+1, row)
			f.SetCellValue(sheetName, cell, turtle.OriginalData[col])
		}

		// Write results for each threshold
		colOffset := len(header)
		for threshold, results := range thresholdResults {
			cell, _ := excelize.CoordinatesToCellName(colOffset+1, row)
			if results[turtle.ID] {
				f.SetCellValue(sheetName, cell, "Yes")
			} else {
				f.SetCellValue(sheetName, cell, "No")
			}
			colOffset++
		}
	}

	// Delete default sheet
	f.DeleteSheet("Sheet1")

	// Save file
	return f.SaveAs(outputPath)
}

// writeCSVOutput writes results to a CSV file
func writeCSVOutput(turtles []Turtle, thresholdResults map[float64]map[string]bool, header []string, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Create header with threshold columns
	newHeader := append([]string{}, header...)
	for threshold := range thresholdResults {
		colName := fmt.Sprintf("near_project_%gkm", threshold)
		newHeader = append(newHeader, colName)
	}

	// Write header
	if err := writer.Write(newHeader); err != nil {
		return err
	}

	// Write data rows
	for _, turtle := range turtles {
		row := make([]string, len(newHeader))

		// Add original data
		for i, col := range header {
			row[i] = turtle.OriginalData[col]
		}

		// Add results for each threshold
		colOffset := len(header)
		for threshold, results := range thresholdResults {
			if results[turtle.ID] {
				row[colOffset] = "Yes"
			} else {
				row[colOffset] = "No"
			}
			colOffset++
		}

		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func main() {
	startTime := time.Now()

	// Load configuration
	config := loadConfig("config.yaml")

	// Validate input files
	if _, err := os.Stat(config.Files.Projects); os.IsNotExist(err) {
		log.Fatalf("Projects file not found: %s", config.Files.Projects)
	}
	if _, err := os.Stat(config.Files.Turtles); os.IsNotExist(err) {
		log.Fatalf("Turtles file not found: %s", config.Files.Turtles)
	}

	fmt.Println("Using files:")
	fmt.Printf("  Projects: %s\n", config.Files.Projects)
	fmt.Printf("  Turtles: %s\n", config.Files.Turtles)
	fmt.Printf("  Output: %s\n", config.Files.Output)

	// Load project data
	projData, _, err := readExcelFile(config.Files.Projects, true)
	if err != nil {
		log.Fatalf("Error reading projects: %s", err)
	}
	projects := projData.([]Project)

	// Load turtle data
	turtleData, turtleHeader, err := readExcelFile(config.Files.Turtles, false)
	if err != nil {
		log.Fatalf("Error reading turtles: %s", err)
	}
	turtles := turtleData.([]Turtle)

	fmt.Printf("Loaded %d projects and %d turtles\n", len(projects), len(turtles))

	// Process each threshold
	thresholdResults := make(map[float64]map[string]bool)

	for _, threshold := range config.Analysis.DistanceThresholds {
		fmt.Printf("Analyzing turtle proximity within %.1fkm of projects...\n", threshold)

		var results map[string]bool

		// Choose best algorithm based on data size and configuration
		large_dataset := len(turtles) > 10000 || len(projects) > 100

		if large_dataset && config.Performance.UseSpatialIndex {
			fmt.Println("Using spatial grid index")
			results = processSpatialGrid(projects, turtles, threshold)
		} else if large_dataset && config.Performance.UseParallel && runtime.NumCPU() > 1 {
			fmt.Printf("Using parallel processing with %d CPUs\n", runtime.NumCPU())
			results = processParallel(projects, turtles, threshold)
		} else {
			fmt.Println("Using sequential processing")
			// Sequential processing
			results = make(map[string]bool)
			for _, project := range projects {
				for _, turtle := range turtles {
					if turtle.ReportDate.Before(project.StartDate) || turtle.ReportDate.After(project.EndDate.AddDate(0, 0, 10)) {
						continue
					}

					distance := haversine(project.Latitude, project.Longitude, turtle.Latitude, turtle.Longitude)
					if distance <= threshold {
						results[turtle.ID] = true
					}
				}
			}
		}

		thresholdResults[threshold] = results

		// Count turtles within this threshold
		count := 0
		for _, v := range results {
			if v {
				count++
			}
		}
		fmt.Printf("Found %d turtles within %.1fkm of projects\n", count, threshold)
	}

	// Write results
	if err := writeResults(turtles, thresholdResults, turtleHeader, config.Files.Output); err != nil {
		log.Fatalf("Error writing results: %s", err)
	}

	// Calculate elapsed time
	elapsed := time.Since(startTime)
	fmt.Printf("Analysis complete in %s\n", elapsed)
	fmt.Printf("Results saved to %s\n", config.Files.Output)
}