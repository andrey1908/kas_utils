#pragma once

#include <string>
#include <chrono>
#include <iterator>
#include <vector>
#include <array>

namespace kas_utils {

extern const std::time_t START_TIME;

template <typename T>
double toSeconds(const T& duration) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

template<typename T>
inline typename std::vector<T>::iterator fastErase(
        std::vector<T>& v, const typename std::vector<T>::iterator& it) {
    if (it == std::prev(v.end())) {
        v.pop_back();
        return v.end();
    }
    *it = v.back();
    v.pop_back();
    return it;
}

template<typename toType, typename fromType, std::size_t N>
inline std::array<toType, N> castArray(const std::array<fromType, N>& fromArray) {
    std::array<toType, N> toArray;
    for (std::size_t i = 0; i < N; i++) {
        toArray[i] = static_cast<toType>(fromArray[i]);
    }
    return toArray;
}

}