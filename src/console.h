#pragma once

#include <iostream>

struct console {
	using color_t = int;

	// Standard ANSI colors
	static constexpr color_t reset = 0;        // Reset
	static constexpr color_t black = 30;      // Black
	static constexpr color_t red = 31;        // Red
	static constexpr color_t green = 32;      // Green
	static constexpr color_t yellow = 33;     // Yellow
	static constexpr color_t blue = 34;       // Blue
	static constexpr color_t magenta = 35;    // Magenta
	static constexpr color_t cyan = 36;       // Cyan
	static constexpr color_t white = 37;      // White

	// Bright ANSI colors
	static constexpr color_t bright_black = 90;   // Bright Black (Gray)
	static constexpr color_t bright_red = 91;     // Bright Red
	static constexpr color_t bright_green = 92;   // Bright Green
	static constexpr color_t bright_yellow = 93;  // Bright Yellow
	static constexpr color_t bright_blue = 94;    // Bright Blue
	static constexpr color_t bright_magenta = 95; // Bright Magenta
	static constexpr color_t bright_cyan = 96;    // Bright Cyan
	static constexpr color_t bright_white = 97;   // Bright White

	template <typename T>
	static std::string format(const T& value) {
		return std::format("{}", value);
	}

	template <typename T>
	static void print(const T& value) {
		std::cout << format(value);
	}

	template <typename FMT, typename... Args> requires std::convertible_to<FMT, std::string_view>
	static std::string format(const FMT& format_str, Args&&... args) {
		return std::vformat(format_str, std::make_format_args(args...));
	}

	template <typename FMT, typename... Args> requires std::convertible_to<FMT, std::string_view>
	static void print(const FMT& format_str, Args&&... args) {
		std::cout << format(format_str, std::forward<Args>(args)...);
	}

	template <typename FMT, typename... Args> requires std::convertible_to<FMT, std::string_view>
	static std::string format(const color_t color, const FMT& format_str, Args&&... args) {
		const std::string formatted = std::vformat(format_str, std::make_format_args(args...));
		return fmt_color(color) + formatted + fmt_color(reset);
	}

	template <typename FMT, typename... Args> requires std::convertible_to<FMT, std::string_view>
	static void print(const color_t color, const FMT& format_str, Args&&... args) {
		std::cout << format(color, format_str, std::forward<Args>(args)...);
	}

private:
	static std::string fmt_color(const color_t color) {
		return std::format("\033[{}m", static_cast<int>(color));
	}
};