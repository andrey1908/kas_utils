#include "kas_utils/dilation.hpp"

#include <stdexcept>


namespace kas_utils {

Dilation::Dilation(int dilation_size) {
    initialize(dilation_size);
}


void Dilation::initialize(int dilation_size) {
    if (dilation_size < 0) {
        throw std::runtime_error("Dilation: got negative dilation size.");
    }

    dilation_size_ = dilation_size;

    if (dilation_size_ > 0) {
        dilation_size_sqr_ = dilation_size_ * dilation_size_;
        dilation_width_ = 2 * dilation_size_ + 1;
        computeDilationPixels();
    }
}


void Dilation::computeDilationPixels() {
    dilation_pixels_.clear();
    dilation_width_to_pixels_num_.clear();
    std::map<int, std::set<int>> row_to_cols;
    int num_pixels = 0;
    for (int y = -dilation_size_; y <= dilation_size_; y++) {
        for (int x = -dilation_size_; x <= dilation_size_; x++) {
            if (y * y + x * x <= dilation_size_sqr_) {
                row_to_cols[y].insert(x);
                num_pixels++;
            }
        }
    }
    dilation_width_to_pixels_num_.push_back(0);
    while (num_pixels > 0) {
        for (auto& [row, cols] : row_to_cols) {
            if (cols.empty()) {
                continue;
            }
            auto lastColIt = std::prev(cols.end());
            dilation_pixels_.push_back(PixelCoords{row, *lastColIt});
            cols.erase(lastColIt);
            num_pixels--;
        }
        dilation_width_to_pixels_num_.push_back(dilation_pixels_.size());
    }
}

}
