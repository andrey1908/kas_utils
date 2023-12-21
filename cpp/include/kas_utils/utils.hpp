#pragma once

#include <string>
#include <chrono>
#include <iterator>
#include <vector>

namespace kas_utils {

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

}