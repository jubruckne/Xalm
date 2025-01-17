#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <regex>
#include <source_location>

constexpr std::string _profile_arg(const std::source_location loc = std::source_location::current()) noexcept {
  return loc.function_name();
}

inline std::string _profile_arg(const std::string& method) noexcept {
  return method;
}

#define profile(...) Profiler __scoped_profiler(_profile_arg(__VA_ARGS__))

class Profiler {
  using clk = std::chrono::high_resolution_clock;
public:
  explicit Profiler(const std::string& method) noexcept : method(method), start_time(clk::now()) {}

  Profiler(const Profiler &other) = delete;
  Profiler(Profiler &&other) noexcept = delete;
  Profiler & operator=(const Profiler &other) = delete;
  Profiler & operator=(Profiler &&other) noexcept = delete;

  ~Profiler() noexcept {
    const auto end_time = clk::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    record(method, duration);
  }

  static void report() {
    std::print("\n");
    std::printf("%-64s %12s %12s\n", "Function", "Calls", "Time (s)");
    std::printf("%s\n", std::string(64 + 12 + 12 + 2, '-').c_str());
    const std::regex cleanupRegex(R"((\b__restrict\b|\bconst\b|\<\>| const$|\s+(?=\*|&)))");
    const std::regex extraSpacesRegex(R"(\s{2,})");

    for (const auto& [name, stats] : profile_data) {
      std::string cleanedName = std::regex_replace(std::regex_replace(name, cleanupRegex, ""), extraSpacesRegex, " ");

      std::printf("%-64s %12i %lld\n",
                  cleanedName.c_str(),
                  stats.call_count,
                  stats.total_time / 1000);
    }
  }

private:
  std::string method;
  std::chrono::time_point<clk> start_time;

  struct ProfileStats {
    long long total_time = 0;
    int call_count = 0;
  };

  static inline std::unordered_map<std::string, ProfileStats> profile_data;
  static inline std::mutex mutex;

  static void record(const std::string& method, const long long duration) noexcept {
    std::lock_guard lock(mutex);
    auto&[total_time, call_count] = profile_data[method];
    total_time += duration;
    call_count += 1;
  }
};
