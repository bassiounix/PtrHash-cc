#ifndef BINARY_HEAP_HPP_
#define BINARY_HEAP_HPP_

#include <vector>
#include <optional>

template <typename T> class BinaryHeap {
private:
  std::vector<T> data;

  void heapify_up(size_t index) {
    while (index > 0) {
      size_t parent = (index - 1) / 2;
      if (data[index] <= data[parent])
        break;
      std::swap(data[index], data[parent]);
      index = parent;
    }
  }

  void heapify_down(size_t index) {
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
  BinaryHeap() = default;

  void push(const T &value) {
    data.push_back(value);
    heapify_up(data.size() - 1);
  }

  void push(T &&value) {
    data.push_back(std::move(value));
    heapify_up(data.size() - 1);
  }

  std::optional<T> pop() {
    if (data.empty())
      return std::nullopt;
    T top = data[0];
    data[0] = data.back();
    data.pop_back();
    if (!data.empty())
      heapify_down(0);
    return top;
  }

  std::optional<const T> peek() const {
    if (data.empty())
      return std::nullopt;
    return data[0];
  }

  size_t size() const { return data.size(); }

  bool empty() const { return data.empty(); }
};

#endif  // BINARY_HEAP_HPP_
