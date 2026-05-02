#include <filesystem>
#include <iostream>
#include <maximus/database_catalogue.hpp>
#include <regex>

namespace fs = std::filesystem;

namespace maximus {
DatabaseCatalogue::DatabaseCatalogue(std::string base_path): base_path_(base_path) {
}

std::vector<std::string> DatabaseCatalogue::table_paths(std::string table_name) const {
    std::vector<std::string> matching_files;

    // std::cout << "[DEBUG] Looking for table fragments: " << table_name << std::endl;
    // std::cout << "[DEBUG] Base path: " << base_path_ << std::endl;

    if (!fs::exists(base_path_)) {
        std::cerr << "[ERROR] Base path does not exist: " << base_path_ << std::endl;
        throw std::runtime_error("Base path does not exist: " + base_path_);
    }

    std::regex pattern("^" + table_name + "(-\\d+)?\\.(csv|parquet)$");
    // std::cout << "[DEBUG] Regex pattern: ^" << table_name << "(-\\d+)?\\.(csv|parquet)$" << std::endl;

    for (const auto& entry : fs::directory_iterator(base_path_)) {
        if (!entry.is_regular_file()) {
            // std::cout << "[DEBUG] Skipping non-file: " << entry.path() << std::endl;
            continue;
        }

        std::string filename = entry.path().filename().string();
        // std::cout << "[DEBUG] Checking file: " << filename << std::endl;

        if (std::regex_match(filename, pattern)) {
            // std::cout << "[DEBUG] MATCH: " << filename << std::endl;
            matching_files.push_back(entry.path().string());
        } else {
            // std::cout << "[DEBUG] NO MATCH: " << filename << std::endl;
        }
    }

    if (matching_files.empty()) {
        std::cerr << "[ERROR] No fragments found for table: " << table_name << std::endl;
        throw std::runtime_error("No fragments found for table: " + table_name);
    }

    std::sort(matching_files.begin(), matching_files.end());
    /* 
    std::cout << "[DEBUG] Matching files found:" << std::endl;
    for (const auto& path : matching_files) {
        std::cout << "    " << path << std::endl;
    }
    */

    return matching_files;
}


/*
std::string DatabaseCatalogue::table_path(std::string table_name) const {
    // Check if a CSV file exists
    std::string full_table_name = table_name;
    std::string csv_file        = base_path_ + "/" + full_table_name + ".csv";
    if (fs::exists(csv_file)) {
        return csv_file;
    }

    // Check if a Parquet file exists
    std::string parquet_file = base_path_ + "/" + full_table_name + ".parquet";
    if (fs::exists(parquet_file)) {
        return parquet_file;
    }

    throw std::runtime_error("Table not found: " + full_table_name);
}
*/

std::shared_ptr<DatabaseCatalogue> make_catalogue(std::string base_path) {
    return std::make_shared<DatabaseCatalogue>(base_path);
}
}  // namespace maximus
