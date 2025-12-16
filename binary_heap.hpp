#ifndef BINARY_HEAP_HPP_
#define BINARY_HEAP_HPP_

#include <vector>

template <typename T> class BinaryHeap {
private:
  mutable std::vector<T> data;

  constexpr void heapify_up(size_t index) const {
    while (index > 0) {
      size_t parent = (index - 1) / 2;
      if (data[index] <= data[parent])
        break;
      std::swap(data[index], data[parent]);
      index = parent;
    }
  }

  constexpr void heapify_down(size_t index) const {
    size_t n = data.size();
    while (true) {
      size_t left = 2 * index + 1;
      size_t right = 2 * index + 2;
      size_t largest = index;

      if (left < n && data[left] > data[largest])
        largest = left;
      if (right < n && data[right] > data[largest])
        largest = right;

      if (largest == index)
        break;

      std::swap(data[index], data[largest]);
      index = largest;
    }
  }

public:
  constexpr BinaryHeap() = default;
  constexpr ~BinaryHeap() = default;

  constexpr void push(const T &value) const {
    data.push_back(value);
    heapify_up(data.size() - 1);
  }

  constexpr void push(T &&value) const {
    data.push_back(std::move(value));
    heapify_up(data.size() - 1);
  }

  constexpr T pop() const {
    // if (data.empty())
    T top = data[0];
    data[0] = data.back();
    data.pop_back();
    if (!data.empty())
      heapify_down(0);
    return top;
  }

  constexpr const T peek() const {
    // if constexpr (data.empty())
    return data[0];
  }

  constexpr size_t size() const { return data.size(); }

  constexpr bool empty() const { return data.empty(); }
};

#endif  // BINARY_HEAP_HPP_
