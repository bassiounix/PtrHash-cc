#ifndef ZIP_HPP_
#define ZIP_HPP_

#include <tuple>

template <typename It1, typename It2>
class zip_iterator {
public:
    using value_type = std::pair<It1, It2>;

    constexpr zip_iterator(It1 it1, It2 it2) : it1_(it1), it2_(it2) {}

    constexpr const zip_iterator& operator++() const {
        ++it1_;
        ++it2_;
        return *this;
    }

    constexpr bool operator!=(const zip_iterator& other) const {
        return it1_ != other.it1_ && it2_ != other.it2_;
    }

    constexpr auto operator*() const {
        return std::tie(*it1_, *it2_);
    }

private:
    mutable It1 it1_;
    mutable It2 it2_;
};

template <typename Container1, typename Container2>
class zip_wrapper {
public:
    constexpr zip_wrapper(Container1& c1, Container2& c2) : c1_(c1), c2_(c2) {}

    constexpr auto begin() const { return zip_iterator(c1_.begin(), c2_.begin()); }
    constexpr auto end() const { return zip_iterator(c1_.end(), c2_.end()); }

private:
    Container1& c1_;
    Container2& c2_;
};

// Helper function
template <typename C1, typename C2>
constexpr auto zip(C1& c1, C2& c2) {
    return zip_wrapper<C1, C2>(c1, c2);
}


#endif  // ZIP_HPP_
