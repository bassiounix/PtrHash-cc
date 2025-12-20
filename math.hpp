#ifndef MATH_HPP_
#define MATH_HPP_

#include <type_traits>

constexpr double constexpr_ln(double x) {
    if (x <= 0.0)
        return 0.0; // or static_assert in C++23

    // Range reduction: bring x to [0.5, 2)
    int k = 0;
    while (x > 2.0) {
        x *= 0.5;
        ++k;
    }
    while (x < 0.5) {
        x *= 2.0;
        --k;
    }

    // ln(x) ≈ 2 * sum_{n=0}^∞ (1/(2n+1)) * ((x-1)/(x+1))^(2n+1)
    const double y = (x - 1.0) / (x + 1.0);
    const double y2 = y * y;

    double term = y;
    double sum = 0.0;

    // 20 iterations ≈ double precision accuracy
    for (int n = 1; n <= 39; n += 2) {
        sum += term / n;
        term *= y2;
    }

    // ln(x) = 2 * sum + k * ln(2)
    return 2.0 * sum + k * 0.693147180559945309417232121458176568;
}

template <typename T>
constexpr T constexpr_ceil(T x) {
    static_assert(std::is_floating_point_v<T>);
    long long i = static_cast<long long>(x);
    return (static_cast<T>(i) == x) ? x
         : (x > T{0} ? static_cast<T>(i + 1) : static_cast<T>(i));
}

#endif  // MATH_HPP_
