#pragma once

#include <iostream>
#include <mutex>
#include <thread>
#include "profiler.h"

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

	static void init() {
		std::locale::global(std::locale(""));
	}

	template <typename T>
	static std::string format(const T& value) {
		return std::format("{}", value);
	}

	static void print() {
		std::cout << "\n";
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

	template<typename FMT, typename... Args>
		requires std::convertible_to<FMT, std::string_view>
	static void print(const color_t color, const FMT& format_str, Args&&... args) {
		std::cout << format(color, format_str, std::forward<Args>(args)...);
	}

	template<typename FMT, typename... Args>
		requires std::convertible_to<FMT, std::string_view>
	static void error(const FMT& format_str, Args&&... args) {
		std::cout << format(bright_red, format_str, std::forward<Args>(args)...);
		exit(1);
	}


private:
	static std::string fmt_color(const color_t color) {
		return std::format("\033[{}m", static_cast<int>(color));
	}
};

class ProgressBar {
public:
    explicit ProgressBar(const int total_steps, const int width = 50) noexcept:
	total_steps_(total_steps),
	steps_completed_(0),
	width_(width),
	running_(true) {
        update_thread_ = std::thread([this] {
			set_thread_affinity();
			this->update();
        });
    }

    // **Rule of 5: Preventing Copy & Assignment**
    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&& other) = delete;
    ProgressBar& operator=(ProgressBar&& other) = delete;

    ~ProgressBar() noexcept {
    	if (running_) done(status_text_);
    	if (update_thread_.joinable()) {
    		update_thread_.join();
    	}
    }

    void step() noexcept {
        std::scoped_lock lock(mutex_);
        steps_completed_ = std::min(steps_completed_ + 1, total_steps_);
    }

	void step(const std::string_view status) noexcept {
    	std::scoped_lock lock(mutex_);
    	status_text_ = status;
    	steps_completed_ = std::min(steps_completed_ + 1, total_steps_);
    }

    void set_steps_completed(const int steps) noexcept {
        std::scoped_lock lock(mutex_);
        steps_completed_ = std::min(steps, total_steps_);
    }

    void set_status(const std::string_view status) noexcept {
        std::scoped_lock lock(mutex_);
        status_text_ = status;
    }

    void done(const std::string_view final_status) noexcept {
    	std::print("done({})", running_);
    	std::flush(std::cout);

    	if (!running_) return;

        running_ = false;
        status_text_ = final_status;

        if (update_thread_.joinable()) {
            update_thread_.join();
        }

    	auto su = system_usage_.measure();

    	std::print(
    		"\r\033[2K{} | wall: {}ms | user: {}ms | system: {}ms\n",
    		render_bar(-1, width_, status_text_), su.wall_time_ms, su.user_time_ms, su.system_time_ms
    	);

    	std::fflush(stdout);
    }

private:
    void update() const noexcept {
        while (running_) {
	        {
		        std::scoped_lock lock(mutex_);

	        	const int progress = (total_steps_ > 0) ? (steps_completed_ * 100) / total_steps_ : 0;

	        	std::print("\r\033[2K{}", render_bar(progress, width_, status_text_));
	        	std::flush(std::cout);
	        }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    [[nodiscard]] static std::string render_bar(const int progress, const int width, const std::string_view status) noexcept {
		const int prog = (progress * width) / 100;
    	std::string filled;
    	std::string empty;
    	for (int i = 0; i < width; i++) {
    		if (i < prog || progress == -1) {
    			filled += "■";
    		} else {
    			empty += "□";
    		}
    	}
    	if (progress >= 0) {
    		return std::format("{:<20} [{}{}] {}%", status, filled, empty, progress);
    	}

    	return std::format("{:<20} [{}{}]", status, filled, empty);
    }

	static void set_thread_affinity() noexcept {
#if defined(__APPLE__)
    	// macOS: Set Quality of Service (QoS) to `UTILITY` for efficiency cores
    	pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#elif defined(__linux__)
    	// Linux: Bind thread to an efficiency core
    	cpu_set_t cpuset;
    	CPU_ZERO(&cpuset);

    	// Assign to first efficiency core (modify if needed)
    	CPU_SET(0, &cpuset);  // Usually, efficiency cores are lower-numbered

    	pthread_t thread = pthread_self();
    	pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
    }

	/*
    [[nodiscard]] std::string animate_status(const std::string& text) const noexcept {
        if (text.empty()) return text;

		std::string animated;
        for (std::size_t i = 0; i < text.size(); ++i) {
			constexpr int min_intensity = 240;
			constexpr int max_intensity = 255;
			int intensity = min_intensity + (std::abs(static_cast<int>(i) - animation_offset_) * (max_intensity - min_intensity) / text.size());
            animated += std::format("\033[38;5;{}m{}\033[0m", intensity, text[i]);
        }
        return animated;
    }

    void update_animation() noexcept {
        if (status_.empty()) return;

        animation_offset_ += animation_direction_;
        if (animation_offset_ >= status_.size() || animation_offset_ == 0) {
            animation_direction_ *= -1;
        }
    }*/

    int total_steps_;
	int steps_completed_;
    int width_;
    bool running_;
    std::string status_text_;
    std::thread update_thread_;
    mutable std::mutex mutex_;
	system_usage system_usage_;
};