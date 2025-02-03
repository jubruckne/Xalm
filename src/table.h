#pragma once

#include <array>
#include <string>
#include <vector>
#include <tuple>
#include <variant>
#include <sstream>
#include <format>
#include <algorithm>
#include <iomanip>
#include <optional>

// ---------------------------------------------------------
// alignment enum (snake_case)
// ---------------------------------------------------------
enum class alignment {
    left,
    right,
    center
};

// ---------------------------------------------------------
// Helper: deduce alignment for type T
//   - If T is arithmetic (int, float, etc.), => left
//   - Otherwise => right
// ---------------------------------------------------------
template<typename T>
constexpr alignment deduce_alignment() {
    if constexpr (std::is_arithmetic_v<T>) {
        return alignment::left; // numbers => left
    } else {
        return alignment::right; // text => right
    }
}

// ---------------------------------------------------------
// Helper: deduce a default format string
//   - If T is floating => "{:.2f}"
//   - Else => "{}"
// ---------------------------------------------------------
template<typename T>
std::string deduce_format_string() {
    if constexpr (std::is_floating_point_v<T>) {
        return "{:.2f}";
    } else {
        return "{}";
    }
}

template<typename T>
concept is_arithmetic_or_array =
        std::is_arithmetic_v<T> ||
        (requires { typename T::value_type; } && requires(T t) { t.size(); });

template<typename T>
concept is_array_type = requires { typename T::value_type; } && requires(T t) { t.size(); };

template<typename T>
concept is_column_type = is_arithmetic_or_array<T> || std::is_assignable_v<T, std::string>;

// ---------------------------------------------------------
// column definition
//   - We now store format_string instead of decimals
//   - alignment can be deduced if the user doesn't supply one
// ---------------------------------------------------------
template<typename T> requires is_column_type<T>
struct column {
    using value_type = T;

    std::string header;
    int width = -1; // -1 => auto-fit
    alignment align = deduce_alignment<T>();
    std::string format_string = deduce_format_string<T>();
    bool separator = false; // printed after each column

    // 1) Constructor that DEDUCES alignment and format string
    //    if the user does not provide them
    explicit column(std::string hdr,
                    const int w = -1,
                    std::optional<std::string> fmt = std::nullopt,
                    const bool sep = false)
        : header(std::move(hdr))
          , width(w)
          , align(deduce_alignment<T>()) // deduce
          , format_string(fmt.value_or(deduce_format_string<T>())) // deduce
          , separator(sep) {
    }

    // 2) Overloaded constructor if the user explicitly wants to set alignment & format
    column(std::string hdr,
           int w,
           alignment a,
           std::string fmt_str = "{}",
           const bool sep = false)
        : header(std::move(hdr))
          , width(w)
          , align(a)
          , format_string(std::move(fmt_str))
          , separator(sep) {
    }
};

// ---------------------------------------------------------
// Check if T is std::array<U, N>
// ---------------------------------------------------------
template<typename T>
struct is_std_array : std::false_type {
};

template<typename U, std::size_t N>
struct is_std_array<std::array<U, N> > : std::true_type {
};

template<typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

// ---------------------------------------------------------
// array_size_v<T> => N if T=std::array<U,N>, else 1
// ---------------------------------------------------------
template<typename T>
struct array_size {
    static constexpr std::size_t value = 1; // default
};

template<typename U, std::size_t N>
struct array_size<std::array<U, N> > {
    static constexpr std::size_t value = N;
};

template<typename T>
constexpr std::size_t array_size_v = array_size<T>::value;

// ---------------------------------------------------------
// row_variant: either data row or separator row
// ---------------------------------------------------------
struct separator_marker {
}; // empty marker

template<typename... Ts>
using row_variant = std::variant<std::tuple<Ts...>, separator_marker>;


// If the string is longer than `width`, we just return it as-is.
inline std::string center_string(const std::string &str, int width) {
    int length = static_cast<int>(str.size());
    if (length >= width) {
        // No centering if the content is already wider than the width
        return str;
    }

    int total_spaces = width - length;
    int left_spaces = total_spaces / 2;
    int right_spaces = total_spaces - left_spaces;

    return std::string(left_spaces, ' ') + str + std::string(right_spaces, ' ');
}

template<typename T>
std::string format_value(const T &value, const std::string &fmt) {
    if (fmt == "{h}") {
        // format as human readable
        // floats with 2 decimals.
        // ints with
        if constexpr (std::is_floating_point_v<T>) {
            return std::vformat("{.2f}", std::make_format_args(value));
        } else if constexpr (std::is_integral_v<T>) {
            auto s = std::format("{}", value); //TODO fixme
            return std::format("{}", s);
        } else {
            return std::format("{}", value);
        }
    }

    return std::vformat(fmt, std::make_format_args(value));
}

// ---------------------------------------------------------
// get_subcolumn_value
// For a scalar T, subcol_index must be 0.
// For an array T=std::array<U,N>, format arr[subcol_index] with col.format_string
// ---------------------------------------------------------
template<typename T>
std::string get_subcolumn_value(const T &value,
                                const column<T> &col_spec,
                                std::size_t subcol_index) {
    static_assert(!is_std_array_v<T>,
                  "get_subcolumn_value(scalar) called on array type!");
    if (subcol_index != 0) {
        throw std::runtime_error("subcol_index != 0 for a non-array column");
    }
    // For a scalar, just call std::format(col_spec.format_string, value)
    return format_value(value, col_spec.format_string);
}

template<typename U, std::size_t N>
std::string get_subcolumn_value(const std::array<U, N> &arr,
                                const column<std::array<U, N> > &col_spec,
                                std::size_t subcol_index) {
    if (subcol_index >= N) {
        throw std::runtime_error("subcol_index out of range for std::array");
    }
    // We'll create a sub-column that copies alignment & format_string from col_spec
    // so we can call std::format with the sub-element:
    column<U> subcol{
        /*hdr*/ "",
        /*width*/ col_spec.width,
        /*align*/ col_spec.align,
        /*format_str*/ col_spec.format_string,
        /*separator*/ col_spec.separator
    };
    // Now format arr[subcol_index] with subcol format string
    return format_value(arr[subcol_index], subcol.format_string);
}

// ---------------------------------------------------------
// get_subcolumn_header
// For a scalar => just the column header
// For an array => "Header[0]", "Header[1]", etc.
// ---------------------------------------------------------
template<typename T>
std::string get_subcolumn_header(const column<T> &col_spec, std::size_t idx) {
    static_assert(!is_std_array_v<T>,
                  "get_subcolumn_header(scalar) called on array type!");
    if (idx != 0) {
        throw std::runtime_error("idx != 0 for a non-array column");
    }
    return col_spec.header;
}

template<typename U, std::size_t N>
std::string get_subcolumn_header(const column<std::array<U, N> > &col_spec,
                                 std::size_t idx) {
    return col_spec.header + "[" + std::to_string(idx) + "]";
}

// ---------------------------------------------------------
// A small struct for each expanded subcolumn
// ---------------------------------------------------------
struct display_column {
    std::size_t col_index;
    std::size_t sub_index;
    int width{};
    alignment align;
    std::string header_text;
    std::string format_string;
    bool separator;
};

// ---------------------------------------------------------
// table_impl<Ts...>: actual table logic
// ---------------------------------------------------------
template<typename... Ts>
class table_impl {
public:
    using columns_tuple = std::tuple<column<Ts>...>;

    // constructor
    explicit table_impl(const column<Ts> &... specs)
        : columns_tuple_(std::make_tuple(specs...)) {
        // Expand columns into display_columns_
        build_display_columns();
        // Compute widths
        compute_column_widths_and_stats();
    }

    void add_separator() {
        rows_.push_back(separator_marker{});
    }

    template<typename ValueType, typename ColumnType>
    auto convert_to_column_type(const ValueType &value) {
        using ExpectedType = typename ColumnType::value_type;

        if constexpr (is_std_array_v<ExpectedType> && requires { typename ExpectedType::value_type; }) {
            // If ExpectedType is std::array and the value is a vector of the same type
            if constexpr (std::is_same_v<ValueType, std::vector<typename ExpectedType::value_type> >) {
                constexpr std::size_t N = array_size_v<ExpectedType>;
                if (value.size() > N) {
                    throw std::runtime_error(std::format("Vector size {} does not match expected array size {}",
                                                         value.size(), N));
                }
                std::array<typename ExpectedType::value_type, N> result;
                std::copy(value.begin(), value.end(), result.begin());
                std::fill(result.begin() + value.size(), result.end(),
                          static_cast<typename ExpectedType::value_type>(-1));
                return result;
            }
        }

        // For scalar types or other non-convertible types, return the value directly if compatible
        if constexpr (std::is_convertible_v<ValueType, ExpectedType>) {
            return static_cast<ExpectedType>(value);
        } else {
            throw std::runtime_error("Type mismatch: Cannot convert value to expected column type");
        }
    }

    template<typename... Values, std::size_t... I>
    auto add_row_impl(std::index_sequence<I...>, const Values &... values) {
        return std::make_tuple(
            convert_to_column_type<
                Values, std::tuple_element_t<I, columns_tuple>
            >(values)...
        );
    }

    template<typename... Values>
    void add(const Values &... values) {
        constexpr std::size_t num_columns = std::tuple_size_v<columns_tuple>;

        // Create the row by converting each value to the expected column type
        auto converted_row = add_row_impl(std::make_index_sequence<num_columns>{}, values...);

        rows_.push_back(converted_row);
    }

    void add_row(const Ts &... values) {
        rows_.push_back(std::make_tuple(values...));
    }

    [[nodiscard]] std::string format(const std::string &title, int rows = -1) const {
        compute_column_widths_and_stats();

        std::ostringstream oss;
        if (!title.empty()) {
            print_title(oss, title);
        } else {
            print_separator(oss, -1);
        }

        print_header(oss);
        print_separator(oss);

        for (auto &rv: rows_) {
            if (std::holds_alternative<separator_marker>(rv)) {
                print_separator(oss);
            } else {
                print_data_row(oss, std::get<std::tuple<Ts...> >(rv));
            }
            if (rows-- == 0) break;
        }
        print_separator(oss, 1);
        return oss.str();
    }

    // Return a string representation
    [[nodiscard]] std::string format(const int rows = -1) const {
        return format("", rows);
    }

private:
    columns_tuple columns_tuple_;
    std::vector<row_variant<Ts...> > rows_;
    // The "expanded" columns for display (including subcolumns for arrays)
    mutable std::vector<display_column> display_columns_;

    // -----------------------------------------------------
    // build_display_columns
    // Expand each column (scalar => 1 subcolumn, array => N subcolumns)
    // -----------------------------------------------------
    void build_display_columns() {
        auto build_col = [this]<typename T0>(auto col_index_c, T0 &col_spec) {
            using col_type = std::decay_t<T0>;
            using actual_value_type = typename col_type::value_type;
            constexpr std::size_t N = array_size_v<actual_value_type>;

            for (size_t i = 0; i < N; i++) {
                display_column dc;
                dc.col_index = col_index_c;
                dc.sub_index = i;
                dc.width = -1; // will compute
                dc.align = col_spec.align;
                dc.format_string = col_spec.format_string;
                if (i < N - 1) {
                    dc.separator = false;
                } else {
                    dc.separator = col_spec.separator;
                }
                dc.header_text = get_subcolumn_header(col_spec, i);

                display_columns_.push_back(dc);
            }
        };

        apply_to_each_column_const_index(columns_tuple_, build_col);
    }

    // -----------------------------------------------------
    // compute_column_widths
    // -----------------------------------------------------
    void compute_column_widths_and_stats() const {
        for (auto &dc: display_columns_) {
            // We'll find the corresponding column<T> in columns_tuple_
            auto set_width_fn = [this, &dc](auto i_c, auto &col_spec) {
                if (i_c == dc.col_index) {
                    if (col_spec.width != -1) {
                        dc.width = col_spec.width;
                    } else {
                        // auto-compute
                        int max_len = static_cast<int>(dc.header_text.size());
                        for (auto &rv: rows_) {
                            if (auto data_ptr = std::get_if<std::tuple<Ts...> >(&rv)) {
                                auto &value = std::get<i_c>(*data_ptr);
                                std::string val_str =
                                        get_subcolumn_value(value, col_spec, dc.sub_index);
                                max_len = std::max<int>(max_len, static_cast<int>(val_str.size()));
                            }
                        }
                        dc.width = max_len;
                    }
                }
            };
            apply_to_each_column_const_index(columns_tuple_, set_width_fn);
        }
    }

    // -----------------------------------------------------
    // print_header
    // -----------------------------------------------------
    void print_header(std::ostringstream &oss) const {
        oss << "│ ";

        for (auto &dc: display_columns_) {
            if (dc.align == alignment::right) {
                oss << std::right << std::setw(dc.width) << dc.header_text;
            } else if (dc.align == alignment::left) {
                oss << std::left << std::setw(dc.width) << dc.header_text;
            } else if (dc.align == alignment::center) {
                // Manually center the string before printing.
                std::string centered = center_string(dc.header_text, dc.width);

                // We can either just print it directly,
                // or ensure we don't apply std::left/std::right again
                oss << centered;
            } else {
                throw std::runtime_error("unrecognized align");
            }

            if (dc.separator) {
                oss << " │ ";
            } else {
                oss << " ";
            }
        }
        oss << "│\n";
    }

    void print_title(std::ostringstream &oss, const std::string &title) const {
        int total_width = 0;
        for (const auto &dc: display_columns_) {
            total_width += dc.width;
            if (dc.separator) total_width += 3; // Account for separators
        }

        total_width += 2;

        // Center the title and print it
        oss << "╭─" << std::format("{:─<{}}", "", total_width) << "─╮\n";
        oss << "│ " << std::left << std::setw(total_width) << title << " │\n";
        oss << "├─" << std::format("{:─<{}}", "", total_width) << "─┤\n";
    }

    // -----------------------------------------------------
    // print_separator (the "-----" row)
    // -----------------------------------------------------
    void print_separator(std::ostringstream &oss, const int top_or_bottom = 0) const {
        if (top_or_bottom == -1) {
            oss << "╭─";
        } else if (top_or_bottom == 1) {
            oss << "╰─";
        } else {
            oss << "│ ";
        }

        for (auto &dc: display_columns_) {
            oss << std::format("{:─<{}}", "", dc.width);

            if (dc.separator) {
                if (dc.col_index < display_columns_.size()) {
                    if (top_or_bottom == 0) {
                        oss << " │ ";
                    } else if (top_or_bottom == -1) {
                        oss << "─┬─";
                    } else {
                        oss << "─┴─";
                    }
                }
            } else if (dc.col_index < display_columns_.size() - 2) {
                oss << "─";
            }
        }

        if (top_or_bottom == -1) {
            oss << "─╮";
        } else if (top_or_bottom == 1) {
            oss << "─╯";
        } else {
            oss << " │";
        }

        oss << "\n";
    }

    // -----------------------------------------------------
    // print_data_row
    // -----------------------------------------------------
    void print_data_row(std::ostringstream &oss,
                        const std::tuple<Ts...> &row_data) const {
        oss << "│ ";

        for (auto &dc: display_columns_) {
            std::string val_str;
            auto fetch_fn = [&dc, &row_data, &val_str](auto i_c, auto &col_spec) {
                if (i_c == dc.col_index) {
                    auto &value = std::get<i_c>(row_data);
                    val_str = get_subcolumn_value(value, col_spec, dc.sub_index);
                }
            };
            apply_to_each_column_const_index(columns_tuple_, fetch_fn);

            if (dc.align == alignment::right) {
                oss << std::right << std::setw(dc.width) << val_str;
            } else if (dc.align == alignment::left) {
                oss << std::left << std::setw(dc.width) << val_str;
            } else if (dc.align == alignment::center) {
                std::string centered = center_string(val_str, dc.width);
                oss << centered;
            }
            if (dc.separator) {
                oss << " │ ";
            } else {
                oss << " ";
            }
        }
        oss << "│\n";
    }

private:
    // -----------------------------------------------------
    // apply_to_each_column_const_index
    // -----------------------------------------------------
    template<typename TupleT, typename Functor>
    static void apply_to_each_column_const_index(const TupleT &t, Functor &&f) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (f(std::integral_constant<std::size_t, I>{}, std::get<I>(t)), ...);
        }(std::make_index_sequence<std::tuple_size_v<TupleT> >{});
    }
};

// ---------------------------------------------------------
// "table" front-end with static "make(...)"
// ---------------------------------------------------------
namespace detail {
    template<typename X>
    struct get_col_value_type;

    template<typename X>
    struct get_col_value_type<column<X> > {
        using type = X;
    };
}

class table {
public:
    template<typename... ColSpecs>
    static auto make(ColSpecs &&... specs) {
        auto tbl = table_impl<typename detail::get_col_value_type<std::decay_t<ColSpecs> >::type...>(
            std::forward<ColSpecs>(specs)...
        );

        return tbl;
    }
};

#ifdef EXAMPLE_MAIN
#include <iostream>

int main()
{
    // Example 1: Let alignment + format be deduced
    //   - int => alignment::left, format="{}"
    //   - double => alignment::left, format="{:.2f}"
    //   - string => alignment::right, format="{}"
    auto tbl1 = table::make(
        column<int>{"Quantity"},         // default: w=-1 => auto fit
        column<double>{"Price"},         // will default to "{:.2f}" + left alignment
        column<std::string>{"Product"}   // will default to "{}" + right alignment
    );
    tbl1.add_row(10, 29.99, "Blue T-Shirt");
    tbl1.add_row(5, 19.50, "Socks");
    tbl1.add_separator();
    tbl1.add_row(1234, 99.9999, "High-Precision Widget");
    tbl1.add_row(42, 0.0, "Free Sample");

    std::cout << "=== Table with deduced alignment & format ===\n";
    std::cout << tbl1.print() << std::endl;

    // Example 2: Overriding alignment/format
    //   - Force alignment::right for int
    //   - Provide a custom format string for double
    auto tbl2 = table::make(
        column<int>{"Qty", -1, alignment::right, "{}", " | "},
        column<double>{"Price", 10, alignment::left, "{:.3f}", " | "},
        column<std::string>{"Product", 12, alignment::left, "{}", " | "}
    );
    tbl2.add_row(10, 29.99, "Blue T-Shirt");
    tbl2.add_row(5, 19.50, "Socks");
    tbl2.add_separator();
    tbl2.add_row(42, 123.45678, "Long Product Name");
    tbl2.add_row(999, 0.0, "MegaDeal");

    std::cout << "=== Table with overridden alignment & format ===\n";
    std::cout << tbl2.print() << std::endl;

    return 0;
}
#endif
