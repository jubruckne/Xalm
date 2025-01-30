#pragma once

#include "table.h"


#include <chrono>
#include <cstdint>
#include <mutex>
#include <regex>
#include <source_location>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <unordered_map>

constexpr std::string _profile_arg(const std::source_location loc = std::source_location::current()) noexcept {
	return loc.function_name();
}

inline std::string _profile_arg(const std::string& method) noexcept { return method; }

#define profile(...) Profiler __scoped_profiler(_profile_arg(__VA_ARGS__))

class Profiler {
	using clk = std::chrono::high_resolution_clock;

public:
	explicit Profiler(const std::string& method) noexcept : method(method), start_time(clk::now()) {}

	Profiler(const Profiler& other) = delete;
	Profiler(Profiler&& other) noexcept = delete;
	Profiler& operator=(const Profiler& other) = delete;
	Profiler& operator=(Profiler&& other) noexcept = delete;

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

		for (const auto& [name, stats]: profile_data) {
			std::string cleanedName =
					std::regex_replace(std::regex_replace(name, cleanupRegex, ""), extraSpacesRegex, " ");

			std::printf("%-64s %12i %lld\n", cleanedName.c_str(), stats.call_count, stats.total_time / 1000);
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
		auto& [total_time, call_count] = profile_data[method];
		total_time += duration;
		call_count += 1;
	}
};

struct system_usage {
private:
	rusage start_usage{};
	std::chrono::steady_clock::time_point start_time;

	static constexpr int64_t timeval_to_ms(const timeval& tv) noexcept {
		// sec -> milliseconds + usec -> milliseconds
		return static_cast<int64_t>(tv.tv_sec) * 1000LL +
		       static_cast<int64_t>(tv.tv_usec) / 1000LL;
	}

public:
	struct scoped;

	struct info {
		int64_t user_time_ms; // User CPU time in milliseconds
		int64_t system_time_ms; // System CPU time in milliseconds
		int64_t wall_time_ms; // Real elapsed time in milliseconds (wall clock)
		int64_t page_faults; // Major page faults (disk I/O required)
		int64_t page_reclaims; // Minor page faults (no disk I/O)
		int64_t swaps; // Number of swaps
		int64_t block_input; // Block input operations
		int64_t block_output; // Block output operations
		int64_t voluntary_ctx_switches; // Voluntary context switches
		int64_t involuntary_ctx_switches; // Involuntary context switches
	};

	void reset() noexcept {
		getrusage(RUSAGE_SELF, &start_usage);
		start_time = std::chrono::steady_clock::now();
	}

	system_usage() noexcept {
		reset();  // Initialize with the first baseline
	}

	// Delete copy and move constructors/assignments
	system_usage(const system_usage&) = delete;
	system_usage(system_usage&&) = delete;
	system_usage& operator=(const system_usage&) = delete;
	system_usage& operator=(system_usage&&) = delete;

	[[nodiscard]] info measure() const noexcept {
		rusage end_usage{};
		const auto end_time = std::chrono::steady_clock::now();
		getrusage(RUSAGE_SELF, &end_usage);

		return {
			timeval_to_ms(end_usage.ru_utime) - timeval_to_ms(start_usage.ru_utime),
			timeval_to_ms(end_usage.ru_stime) - timeval_to_ms(start_usage.ru_stime),
			std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count(),
			end_usage.ru_majflt - start_usage.ru_majflt,
			end_usage.ru_minflt - start_usage.ru_minflt,
			end_usage.ru_nswap - start_usage.ru_nswap,
			end_usage.ru_inblock - start_usage.ru_inblock,
			end_usage.ru_oublock - start_usage.ru_oublock,
			end_usage.ru_nvcsw - start_usage.ru_nvcsw,
			end_usage.ru_nivcsw - start_usage.ru_nivcsw
		    };
	}
};

template <>
struct std::formatter<system_usage::info> {
	constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

	template <typename FormatContext>
	auto format(const system_usage::info& usage, FormatContext& ctx) const {
		auto tbl = table::make(column<std::string>{"performance metrics", -1, alignment::left, "{}", true},
								   column<uint64_t>{"value", -1, alignment::right, "{h}", false});

		tbl.add("user time", usage.user_time_ms);
		tbl.add("system time", usage.system_time_ms);
		tbl.add("wall-time", usage.wall_time_ms);
		tbl.add_separator();
		tbl.add("major page-faults", usage.page_faults);
		tbl.add("minor page-faults", usage.page_reclaims);
		tbl.add("swaps", usage.swaps);
		tbl.add_separator();
		tbl.add("block-input", usage.block_input);
		tbl.add("block-output", usage.block_output);
		tbl.add_separator();
		tbl.add("voluntary ctx switches", usage.voluntary_ctx_switches);
		tbl.add("involuntary ctx switches", usage.involuntary_ctx_switches);
		return std::format_to(ctx.out(), "{}", tbl.format());
	}
};

struct system_usage::scoped {
	explicit scoped(std::string label) noexcept : label(std::move(label)) {}

	~scoped() {
		system_usage::info result = usage.measure();
        std::cout << std::format("{}:\n{}\n", label, result);
	}

private:
	system_usage usage;
	std::string label;
};
