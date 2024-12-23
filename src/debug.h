#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <source_location>

#define profile() Profiler __scoped_profiler

class Profiler {
public:
  Profiler(const std::source_location& location = std::source_location::current())
      : functionName(location.function_name()),
        startTime(std::chrono::high_resolution_clock::now()) {}

  ~Profiler() {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    record(functionName, duration);
  }

  static void report() {
    std::lock_guard<std::mutex> lock(dataMutex);
    std::cout << "Profiling Report:\n";
    for (const auto& [name, stats] : profileData) {
      const double totalMs = stats.totalTime / 1000.0;         // Convert to milliseconds
      const double avgMs = totalMs / stats.callCount;         // Average time in ms
      printf("  %s:\n", name.c_str());
      printf("    Total Time: %0.0f ms\n", totalMs);
      printf("    Calls: %d\n", stats.callCount);
      printf("    Average Time: %0.0f ms\n", avgMs);
    }
  }

private:
  std::string functionName;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

  struct ProfileStats {
    long long totalTime = 0;
    int callCount = 0;
  };

  static inline std::unordered_map<std::string, ProfileStats> profileData;
  static inline std::mutex dataMutex;

  static void record(const std::string& functionName, const long long duration) {
    std::lock_guard lock(dataMutex);
    auto&[totalTime, callCount] = profileData[functionName];
    totalTime += duration;
    callCount += 1;
  }
};

struct BinaryDumper {
  // Save T array to binary file
  template <typename T>
  static bool save(const std::string& filename, const T* data, size_t count);
  
  // Load T array from binary file
  template <typename T>
  static std::vector<T> load(const std::string& filename);
};