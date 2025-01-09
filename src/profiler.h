#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <regex>
#include <source_location>

//#define profile(...) Profiler __scoped_profiler(__VA_ARGS__)
#define profile(...) Profiler __scoped_profiler(std::source_location::current(), ##__VA_ARGS__)

class Profiler {
  using clk = std::chrono::high_resolution_clock;
public:
  explicit Profiler(const std::source_location& location = std::source_location::current(),
    const std::string& params = std::string()) noexcept
      : functionName(location.function_name()), parameters(params),
        startTime(clk::now()) {
  }

  Profiler(const Profiler &other) = delete;
  Profiler(Profiler &&other) noexcept = delete;
  Profiler & operator=(const Profiler &other) = delete;
  Profiler & operator=(Profiler &&other) noexcept = delete;

  ~Profiler() noexcept {
    const auto endTime = clk::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    record(functionName + "<" + parameters + ">", duration);
  }

  static void report() {
    std::printf("\n");
    std::printf("%-64s %12s %12s\n", "Function", "Calls", "Time (s)");
    std::printf("%s\n", std::string(64 + 12 + 12 + 2, '-').c_str());
    const std::regex cleanupRegex(R"((\b__restrict\b|\bconst\b|\<\>| const$|\s+(?=\*|&)))");
    const std::regex extraSpacesRegex(R"(\s{2,})");

    for (const auto& [name, stats] : profileData) {
      std::string cleanedName = std::regex_replace(std::regex_replace(name, cleanupRegex, ""), extraSpacesRegex, " ");

      std::printf("%-64s %12i %12.0f\n",
                  cleanedName.c_str(),
                  stats.callCount,
                  static_cast<double>(stats.totalTime));
    }
  }

private:
  const std::string functionName;
  const std::string parameters;
  std::chrono::time_point<clk> startTime;

  struct ProfileStats {
    long long totalTime = 0;
    int callCount = 0;
  };

  static inline std::unordered_map<std::string, ProfileStats> profileData;
  static inline std::mutex dataMutex;

  static void record(const std::string& functionName, const long long duration) noexcept {
    std::lock_guard lock(dataMutex);
    auto&[totalTime, callCount] = profileData[functionName];
    totalTime += duration;
    callCount += 1;
  }
};
