//
// Created by lwilkinson on 8/20/21.
//

#include "csv_log_io.h"

#include <assert.h>
#include <vector>
#include <iostream>
#include <set>

namespace cpp_testbed {

//
//  Utils
//

template <typename T>
std::string to_string_with_precision(const T a_value, const int n)
{
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

typedef std::map<std::string, std::pair<std::ofstream, std::set<std::string>>> csv_files_cache_t;
static csv_files_cache_t  csv_files_cache;
static std::set<std::string> csv_files_append;


void add_missing_columns(std::map<std::string, csv_row_t>& rows) {
    std::vector<std::string> keys;
    for (const auto& [method, row] : rows) {
        for (const auto &[key, _]: row) {
            keys.push_back(key);
        }
    }

    auto column_names = std::set(keys.begin(), keys.end());
    for (auto& [method, row] : rows) {
        for (const auto &name: column_names) {
            if (row.find(name) == row.end()) {
                csv_row_insert(row, name, "");
            }
        }
    }
}
void mark_file_for_append(std::string filepath) {
    csv_files_append.insert(filepath);
}

void write_csv_rows(
    std::string filepath,
    const std::map<std::string, csv_row_t>& rows
) {
    for (auto& [method, row] : rows) { write_csv_row(filepath, row); }
}

void write_csv_row(
    std::string filepath,
    const csv_row_t& row
) {
  auto file_cache = csv_files_cache.find(filepath);
  bool write_header = false;

  std::vector<std::string> keys;
  for (const auto& [key, _] : row) { keys.push_back(key); }
  auto column_names = std::set(keys.begin(), keys.end());

  if (file_cache == csv_files_cache.end()) {
    bool append = csv_files_append.find(filepath) != csv_files_append.end();

    // First time we have seen this file so we need to open up a `ofstream` and initialize the column names
    csv_files_cache[filepath] = std::make_pair(
      std::ofstream(filepath, append ? std::ios_base::app : std::ios_base::out),
      std::move(column_names)
    );
    file_cache = csv_files_cache.find(filepath);
    write_header = !append;
  } else {
    auto& column_names_written = file_cache->second.second;
    assert(column_names_written == column_names && "Column names do not match the first row written");
  }

  auto& file = file_cache->second.first;

  if (write_header) {
    for (auto it = row.cbegin(); it != row.cend(); ++it)
      file << it->first << ",";
    file << "\n";
  }

  for (auto it = row.cbegin(); it != row.cend(); ++it)
    file << it->second << ",";
  file << "\n";

  file.flush();
}

void csv_row_insert(csv_row_t &csv_row, const std::string& name, double value) {
  csv_row[name] = to_string_with_precision(value);
}

void csv_row_insert(csv_row_t &csv_row, const std::string& name, int value) {
  csv_row[name] = std::to_string(value);
}

//void csv_row_insert(csv_row_t &csv_row, const std::string& name, std::ptrdiff_t value) {
//  csv_row[name] = std::to_string(value);
//}

void csv_row_insert(csv_row_t &csv_row, const std::string& name, size_t value) {
  csv_row[name] = std::to_string(value);
}

void csv_row_insert(csv_row_t &csv_row, const std::string& name, long long value) {
  csv_row[name] = std::to_string(value);
}

void csv_row_insert(csv_row_t &csv_row, const std::string& name, const std::string& value) {
  csv_row[name] = value;
}

} // namespace test_harness