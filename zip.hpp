#ifndef ZIP_HPP_
#define ZIP_HPP_

#include <tuple>

template <typename It1, typename It2>
class zip_iterator {
public:
    using value_type = std::pair<It1, It2>;

    zip_iterator(It1 it1, It2 it2) : it1_(it1), it2_(it2) {}

    zip_iterator& operator++() {
        ++it1_;
        ++it2_;
        return *this;
    }

    bool operator!=(const zip_iterator& other) const {
        return it1_ != other.it1_ && it2_ != other.it2_;
    }

    auto operator*() const {
        return std::tie(*it1_, *it2_);
    }

private:
    It1 it1_;
    It2 it2_;
};

template <typename Container1, typename Container2>
class zip_wrapper {
public:
    zip_wrapper(Container1& c1, Container2& c2) : c1_(c1), c2_(c2) {}

    auto begin() { return zip_iterator(c1_.begin(), c2_.begin()); }
    auto end() { return zip_iterator(c1_.end(), c2_.end()); }

private:
    Container1& c1_;
    Container2& c2_;
};

// Helper function
template <typename C1, typename C2>
auto zip(C1& c1, C2& c2) {
    return zip_wrapper<C1, C2>(c1, c2);
}


#endif  // ZIP_HPP_
