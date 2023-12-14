#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>


namespace kas_utils {

class Dilation {
private:
    struct PixelCoords {
        bool inFrame(int h, int w) const {
            return y >= 0 && x >= 0 && y < h && x < w;
        }
        int y;
        int x;
    };

public:
    Dilation(int dilation_size);
    void initialize(int dilation_size);

    template <typename T, size_t size>
    cv::Mat dilate(const cv::Mat& image, const T (&background_colors)[size],
        bool dilate_background = false) const;

    int dilation_size() const
    {
        return dilation_size_;
    }

private:
    void computeDilationPixels();

private:
    int dilation_size_;
    int dilation_size_sqr_;
    int dilation_width_;

    std::vector<PixelCoords> dilation_pixels_;
    std::vector<int> dilation_width_to_pixels_num_;
};


template <typename T, size_t size>
cv::Mat Dilation::dilate(const cv::Mat& image,
        const T(&background_colors)[size], bool dilate_background /* false */) const {
    if (dilation_size_ == 0) {
        throw std::runtime_error("Dilation: dilate() function was called when dilation size is zero.");
    }

    cv::Mat dilated = image.clone();
    for (int y = 0; y < image.rows; y++) {
        int current_dilation_width = dilation_width_;
        for (int x = 0; x < image.cols; x++) {
            current_dilation_width =
                std::min(current_dilation_width + 1, dilation_width_);
            const T& color = image.at<T>(y, x);
            bool color_is_background =
                std::find(std::begin(background_colors),
                    std::end(background_colors), color) !=
                std::end(background_colors);
            if (color_is_background != dilate_background) {
                continue;
            }
            for (int i = 0; i < dilation_width_to_pixels_num_[current_dilation_width]; i++) {
                PixelCoords pixel_coords = dilation_pixels_[i];
                pixel_coords.y += y;
                pixel_coords.x += x;
                if (!pixel_coords.inFrame(dilated.rows, dilated.cols)) {
                    continue;
                }
                dilated.at<T>(pixel_coords.y, pixel_coords.x) = color;
            }
            current_dilation_width = 0;
        }
    }
    return dilated;
}

}
