#ifndef EXPECTED_HPP_
#define EXPECTED_HPP_

namespace cpp {

// This is used to hold an unexpected value so that a different constructor is
// selected.
template <class T> class unexpected {
  T value;

public:
  inline constexpr explicit unexpected(T value) : value(value) {}
  inline constexpr T error() { return value; }
};

template <class T> explicit unexpected(T) -> unexpected<T>;

template <class T, class E> class expected {
  union {
    T exp;
    E unexp;
  };
  bool is_expected;

public:
  inline constexpr expected(T exp) : exp(exp), is_expected(true) {}
  inline constexpr expected(unexpected<E> unexp)
      : unexp(unexp.error()), is_expected(false) {}

  inline constexpr bool has_value() const { return is_expected; }

  inline constexpr T &value() { return exp; }
  inline constexpr E &error() { return unexp; }
  inline constexpr const T &value() const { return exp; }
  inline constexpr const E &error() const { return unexp; }

  inline constexpr operator bool() const { return is_expected; }

  inline constexpr T &operator*() { return exp; }
  inline constexpr const T &operator*() const { return exp; }
  inline constexpr T *operator->() { return &exp; }
  inline constexpr const T *operator->() const { return &exp; }
};

} // namespace std

#endif // EXPECTED_HPP_
