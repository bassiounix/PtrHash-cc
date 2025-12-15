#ifndef ENUMERATE_HPP_
#define ENUMERATE_HPP_

#include <cstddef>
#include <utility>

template <typename Iterable>
struct Enumerate {
    Iterable& iterable;

    struct Iterator {
        size_t index;
        decltype(std::begin(iterable)) it;

        auto operator*() const {
            return std::pair<size_t, decltype(*it)>(index, *it);
        }

        Iterator& operator++() {
            ++index;
            ++it;
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return it != other.it;
        }
    };

    Iterator begin() const {
        return {0, std::begin(iterable)};
    }

    Iterator end() const {
        return {0, std::end(iterable)};
    }
};


template <typename Iterable>
Enumerate<Iterable> enumerate(Iterable& iterable) {
    return {iterable};
}

template <typename Iterable>
Enumerate<Iterable> enumerate(Iterable&& iterable) {
    return {iterable};
}


#endif  // ENUMERATE_HPP_
