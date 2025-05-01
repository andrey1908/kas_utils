#pragma once

#include <array>
#include <limits>
#include <algorithm>
#include <stdexcept>


namespace kas_utils {

template <typename T, std::size_t Dims>
class MultiArray {
private:
    template <typename MultiArrayType, typename DataType>
    class BlockImpl {
    protected:
        class IteratorImpl {
        private:
            typedef BlockImpl<MultiArrayType, DataType> BlockType;

        public:
            IteratorImpl(const BlockType& block, const std::array<std::size_t, Dims>& position) :
                    startIndex_(block.startIndex_), blockShape_(block.blockShape_),
                    array_(block.array_), position_(position) {
                std::size_t offset = array_.getOffset(startIndex_);
                pointer_ = array_.data_ + offset;
            }

            IteratorImpl(const BlockType& block) :
                    startIndex_(block.startIndex_), blockShape_(block.blockShape_),
                    array_(block.array_), pointer_(nullptr) {}  // end-iterator

            IteratorImpl& operator++ () {
                std::size_t d = 0;
                for (; d < Dims; d++) {
                    std::size_t revD = Dims - 1 - d;
                    std::size_t& pos = position_[revD];
                    pos++;
                    pointer_ += array_.strides_[revD];

                    if (pos == blockShape_[revD]) {
                        pos = 0;
                        pointer_ -= array_.strides_[revD] * blockShape_[revD];
                    } else {
                        break;
                    }
                }
                if (d == Dims) {
                    pointer_ = nullptr;
                }
                return *this;
            }

            const std::array<std::size_t, Dims>& position() const {
                return position_;
            }

            bool operator== (const IteratorImpl& other) const {
                return pointer_ == other.pointer_;
            }

            bool operator!= (const IteratorImpl& other) const {
                return !operator==(other);
            }

            DataType& operator* () const {
                return *pointer_;
            }

        private:
            const std::array<std::size_t, Dims> startIndex_;
            const std::array<std::size_t, Dims> blockShape_;
            const MultiArrayType& array_;

            std::array<std::size_t, Dims> position_;
            DataType* pointer_;
        };

    public:
        BlockImpl(const std::array<std::size_t, Dims>& startIndex, const std::array<std::size_t, Dims>& blockShape,
                const MultiArrayType& array) :
                startIndex_(startIndex), blockShape_(blockShape), array_(array) {
            for (std::size_t d = 0; d < Dims; d++) {
                if (blockShape_[d] == 0) {
                    throw std::runtime_error("MultiArray: creating a block of zero size");
                }
                if (startIndex_[d] + blockShape_[d] > array_.shape()[d]) {
                    throw std::runtime_error("MultiArray: creating a block that exceeds array limits");
                }
            }
        }

        const std::array<std::size_t, Dims>& startIndex() const {
            return startIndex_;
        }

        const std::array<std::size_t, Dims>& blockShape() const {
            return blockShape_;
        }

        MultiArrayType& array() const {
            return array_;
        }

        IteratorImpl begin() const {
            return IteratorImpl(*this, {});
        }

        IteratorImpl end() const {
            return IteratorImpl(*this);
        }

    protected:
        const std::array<std::size_t, Dims> startIndex_;
        const std::array<std::size_t, Dims> blockShape_;
        const MultiArrayType& array_;
    };

    typedef BlockImpl<const MultiArray<T, Dims>, const T> ConstBlock;

    class Block : public BlockImpl<MultiArray<T, Dims>, T> {
    private:
        typedef typename BlockImpl<MultiArray<T, Dims>, T>::IteratorImpl Iterator;

    public:
        Block(const std::array<std::size_t, Dims>& startIndex, const std::array<std::size_t, Dims>& blockShape,
            MultiArray<T, Dims>& array) : BlockImpl<MultiArray<T, Dims>, T>(startIndex, blockShape, array) {}

        void setConstant(const T& value) {
            auto it = begin();
            auto e = end();
            while (it != e) {
                *it = value;
                ++it;
            }
        }

        template <typename BlockType>
        Block& operator= (BlockType&& other) {
            if (!std::equal(this->blockShape_.begin(), this->blockShape_.end(), other.blockShape().begin())) {
                throw std::runtime_error("MultiArray: assigning a block with another block of different size");
            }

            auto it = begin();
            auto otherIt = other.begin();
            auto e = end();
            while (it != e) {
                *it = *otherIt;
                ++it;
                ++otherIt;
            }
            return *this;
        }

        Iterator begin() const {
            return Iterator(*this, {});
        }

        Iterator end() const {
            return Iterator(*this);
        }
    };

    template <typename MultiArrayType, typename DataType>
    class IteratorImpl {
    public:
        IteratorImpl(DataType* pointer, const MultiArrayType& array) : pointer_(pointer), array_(array) {}

        IteratorImpl(const MultiArrayType& array) : array_(array) {
            pointer_ = array_.data_ + array_.size_;
        }  // end-iterator

        IteratorImpl<MultiArrayType, DataType>& operator++ () {
            pointer_++;
            return *this;
        }

        bool operator== (const IteratorImpl<MultiArrayType, DataType>& other) const {
            return pointer_ == other.pointer_;
        }

        bool operator!= (const IteratorImpl<MultiArrayType, DataType>& other) const {
            return !operator==(other);
        }

        DataType& operator* () const {
            return *pointer_;
        }

    protected:
        DataType* pointer_;
        const MultiArrayType& array_;
    };

    typedef IteratorImpl<const MultiArray<T, Dims>, const T> ConstIterator;
    typedef IteratorImpl<MultiArray<T, Dims>, T> Iterator;

public:
    MultiArray() : shape_({}), strides_({}), size_(0), data_(nullptr) {}
    MultiArray(const std::array<std::size_t, Dims>& shape) : data_(nullptr) {
        reset(shape);
    }
    ~MultiArray() {
        delete[] data_;
    }

    void reset() {
        delete[] data_;

        for (std::size_t d = 0; d < Dims; d++) {
            shape_[d] = 0;
            strides_[d] = 0;
        }
        size_ = 0;
    }

    void reset(const std::array<std::size_t, Dims>& newShape) {
        delete[] data_;

        shape_ = newShape;
        std::size_t stride = 1;
        for (std::size_t d = 0; d < Dims; d++) {
            std::size_t revD = Dims - 1 - d;
            strides_[revD] = stride;
            stride *= shape_[revD];
        }
        size_ = stride;

        data_ = new T[size_];
    }

    std::size_t getOffset(const std::array<std::size_t, Dims>& index) const {
        std::size_t offset = 0;
        for (std::size_t d = 0; d < Dims; d++) {
            offset += strides_[d] * index[d];
        }
        return offset;
    }

    T& operator[] (const std::array<std::size_t, Dims>& index) {
        return data_[getOffset(index)];
    }

    const T& operator[] (const std::array<std::size_t, Dims>& index) const {
        return data_[getOffset(index)];
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    void setConstant(const T& value) {
        T* ptr = data_;
        for (std::size_t i = 0; i < size_; i++) {
            *ptr = value;
            ptr++;
        }
    }

    MultiArray<T, Dims>& operator=(const MultiArray<T, Dims>& other) {
        if (data_ != nullptr && !std::equal(shape_.begin(), shape_.end(), other.shape_.begin())) {
            throw std::runtime_error("MultiArray: assigning non-empty array with another array of different size");
        }
        if (data_ == nullptr) {
            reset(other.shape_);
        }

        T* ptr = data_;
        const T* otherPtr = other.data_;
        for (std::size_t i = 0; i < size_; i++) {
            *ptr = *otherPtr;
            ptr++;
            otherPtr++;
        }
        return *this;
    }

    Block block(const std::array<std::size_t, Dims>& startIndex, const std::array<std::size_t, Dims>& blockShape) {
        return Block(startIndex, blockShape, *this);
    }

    ConstBlock block(const std::array<std::size_t, Dims>& startIndex, const std::array<std::size_t, Dims>& blockShape) const {
        return ConstBlock(startIndex, blockShape, *this);
    }

    Iterator begin() {
        return Iterator(data_, *this);
    }

    Iterator end() {
        return Iterator(*this);
    }

    ConstIterator begin() const {
        return ConstIterator(data_, *this);
    }

    ConstIterator end() const {
        return ConstIterator(*this);
    }

    const std::array<std::size_t, Dims>& shape() const {
        return shape_;
    }

    const std::array<std::size_t, Dims>& strides() const {
        return strides_;
    }

    std::size_t size() const {
        return size_;
    }

private:
    std::array<std::size_t, Dims> shape_;
    std::array<std::size_t, Dims> strides_;
    std::size_t size_;

    T* data_;
};

}
