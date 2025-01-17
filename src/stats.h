#pragma once

#include <vector>
#include <algorithm> // for std::min_element, std::max_element
#include <cmath>     // for std::floor

#include "console.h"

namespace stats {
    struct histogram_t {
        std::vector<size_t> bins;
        float32_t bin_width;
        float32_t sum;
        float32_t min;
        float32_t max;
        float32_t mean;
        float32_t range;
        size_t count;

        histogram_t() = delete;
        histogram_t(histogram_t&&) noexcept = default;
        histogram_t& operator=(histogram_t&&) noexcept = default;
        ~histogram_t() = default;

        float32_t calculate_scale() const { return 2.0f / range; }
        float32_t calculate_offset() const { return -(max + min) / 2.0f; }

        explicit histogram_t(const uint8_t bins): bins(std::vector<size_t>(bins)),
                                                  bin_width(0),
                                                  sum(0),
                                                  min(std::numeric_limits<float32_t>::max()),
                                                  max(std::numeric_limits<float32_t>::lowest()),
                                                  mean(0),
                                                  range(0),
                                                  count(0) {
        }

        std::string format() {
            constexpr std::array blocks = {" ", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"};
            constexpr size_t block_levels = blocks.size();

            const size_t max_bin = *std::ranges::max_element(bins);
            const size_t min_bin = *std::ranges::min_element(bins);

            std::string bin_visual = "";
            for (size_t i = 0; i < bins.size(); ++i) {
                const float norm = static_cast<float>(bins[i] - min_bin) / (static_cast<float>(max_bin) - min_bin);
                const size_t block_index = std::ranges::clamp(static_cast<size_t>(std::floor(norm * (block_levels - 1))), 0zu, block_levels - 1);
                assert(block_index < blocks.size() && "Block index out of bounds");
                bin_visual += blocks[block_index];
            }
            return "[" + bin_visual + "]"; // + std::format("{}", bins);
        }
    };

    /**
     * @brief Generates a histogram from the given data using the specified number of bins.
     *
     * @param data  The input data as a std::vector<float>.
     * @param bins  The desired number of bins (must be > 0).
     * @return      A std::vector<int> whose length == bins, containing the counts for each bin.
     *
     * @note        If you need additional information, such as minimum value, maximum value, bin width,
     *              or bin edges, you can either:
     *              1) Return a custom struct that includes them, or
     *              2) Provide additional output parameters.
     */
    static histogram_t histogram(const std::vector<float32_t> &data, const uint8_t bins) {
        histogram_t hist(bins);

        for (const auto& value : data) {
            hist.sum += value;
            hist.min = std::min(hist.min, value);
            hist.max = std::max(hist.max, value);
        }

        hist.range = hist.max - hist.min;
        //std::print("size:{} min {}, max{}, sum{}, range: {}\n", data.size(), hist.min, hist.max, hist.sum, hist.range);

        hist.count = data.size();
        hist.bin_width = hist.range / static_cast<float>(bins);

        for (const float32_t value: data) {
            const int bin_index = static_cast<int>(std::floor((value - hist.min) / hist.range * bins));
            if (bin_index >= bins) {
                ++hist.bins[bins - 1];
            } else {
                ++hist.bins[bin_index];
            }
        }

        hist.mean = hist.sum / hist.count;

        return hist;
    }
}
