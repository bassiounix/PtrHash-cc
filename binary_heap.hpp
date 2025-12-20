#ifndef BINARY_HEAP_HPP_
#define BINARY_HEAP_HPP_

#include <cstddef>
#include <utility>

template <typename T, std::size_t MaxSize = 5> class BinaryHeap {
private:
  mutable T data[MaxSize]{};
  mutable size_t current_size = 0;

  constexpr void heapify_up(std::size_t index) const {
    while (index > 0) {
      std::size_t parent = (index - 1) / 2;
      if (data[index] <= data[parent])
        break;
      std::swap(data[index], data[parent]);
      index = parent;
    }
  }

  constexpr void heapify_down(std::size_t index) const {
    while (true) {
      std::size_t left = 2 * index + 1;
      std::size_t right = 2 * index + 2;
      std::size_t largest = index;

      if (left < current_size && data[left] > data[largest])
        largest = left;
      if (right < current_size && data[right] > data[largest])
        largest = right;

      if (largest == index)
        break;

      std::swap(data[index], data[largest]);
      index = largest;
    }
  }

public:
  constexpr BinaryHeap() = default;

  constexpr void push(const T &value) const {
    if (current_size >= MaxSize)
      return; // Optional: handle overflow
    data[current_size] = value;
    heapify_up(current_size);
    ++current_size;
  }

  constexpr void push(T &&value) const {
    if (current_size >= MaxSize)
      return;
    data[current_size] = std::move(value);
    heapify_up(current_size);
    ++current_size;
  }

  constexpr T pop() const {
    if (current_size == 0)
      return T{}; // Optional: handle underflow
    T top = data[0];
    data[0] = data[current_size - 1];
    --current_size;
    if (current_size > 0)
      heapify_down(0);
    return top;
  }

  constexpr const T &peek() const { return data[0]; }

  constexpr size_t size() const { return current_size; }

  constexpr bool empty() const { return current_size == 0; }

  constexpr bool full() const { return current_size == MaxSize; }
};

#endif // BINARY_HEAP_HPP_
