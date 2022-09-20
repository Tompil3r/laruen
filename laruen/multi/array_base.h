
#ifndef LARUEN_MULTI_ARRAY_BASE_H_
#define LARUEN_MULTI_ARRAY_BASE_H_

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>
#include <ostream>
#include "laruen/multi/types.h"

namespace laruen::multi {

    class ArrayBase {
        template <typename> friend class NDArray;
        template <typename> friend struct NDIter;

        protected:
            // *** member variables are mutable to allow editing of "view"s ***
            mutable Shape shape_;
            mutable Strides strides_;
            mutable Strides dim_sizes_;
            mutable uint_fast64_t size_;
            mutable uint_fast8_t ndim_;
            mutable bool contig_;


        public:
            ArrayBase() noexcept
            : size_(0), ndim_(0), contig_(true)
            {}

            ArrayBase(const Shape &shape, const Strides &strides, const Strides &dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig) noexcept
            : shape_(shape), strides_(strides), dim_sizes_(dim_sizes),
            size_(size), ndim_(ndim), contig_(contig)
            {}

            ArrayBase(Shape &&shape, Strides &&strides, Strides &&dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig) noexcept
            : shape_(std::move(shape)), strides_(std::move(strides)),
            dim_sizes_(std::move(dim_sizes)), size_(size), ndim_(ndim), contig_(contig)
            {}

            ArrayBase(uint_fast8_t ndim, uint_fast64_t size = 0, bool contig = true) noexcept
            : shape_(ndim), strides_(ndim), dim_sizes_(ndim), size_(size), ndim_(ndim), contig_(contig)
            {}

            ArrayBase(Shape::const_iterator begin, Shape::const_iterator end) noexcept
            : shape_(begin, end), strides_(this->shape_.size()), dim_sizes_(this->shape_.size()),
            ndim_(this->shape_.size()), contig_(true)
            {
                uint_fast64_t stride = 1;
                this->size_ = (this->ndim_ > 0);
                
                for(uint_fast8_t dim = this->ndim_; dim-- > 0;) {
                    this->strides_[dim] = stride;
                    stride *= this->shape_[dim];
                    this->dim_sizes_[dim] = stride;
                    this->size_ *= this->shape_[dim];
                }
            }

            explicit ArrayBase(const Shape &shape) noexcept
            : ArrayBase(shape.cbegin(), shape.cend())
            {}

            explicit ArrayBase(Shape &&shape) noexcept
            : shape_(std::move(shape)), strides_(this->shape_.size()), dim_sizes_(this->shape_.size()),
            ndim_(this->shape_.size()), contig_(true)
            {
                uint_fast64_t stride = 1;
                this->size_ = (this->ndim_ > 0);
                
                for(uint_fast8_t dim = this->ndim_; dim-- > 0;) {
                    this->strides_[dim] = stride;
                    stride *= this->shape_[dim];
                    this->dim_sizes_[dim] = stride;
                    this->size_ *= this->shape_[dim];
                }
            }

            ArrayBase(const ArrayBase &base, const Axes &axes, bool contig = false) noexcept
            : shape_(axes.size()), strides_(axes.size()), dim_sizes_(axes.size()),
            size_(axes.size() > 0), ndim_(axes.size()), contig_(contig)
            {
                for(uint_fast8_t i = 0;i < this->ndim_;i++) {
                    uint_fast8_t axis = axes[i];
                    this->shape_[i] = base.shape_[axis];
                    this->strides_[i] = base.strides_[axis];
                    this->dim_sizes_[i] = base.dim_sizes_[axis];
                    this->size_ *= this->shape_[i];
                }
            }

            virtual void reshape(const Shape &shape) {
                if(!this->contig_) {
                    throw std::invalid_argument("invalid operation - non contiguous array cannot be reshaped");
                }

                uint_fast64_t prev_size = this->size_;
                this->ndim_ = shape.size();
                this->shape_ = shape;
                this->strides_.resize(this->ndim_);
                this->dim_sizes_.resize(this->ndim_);
                this->size_ = (this->ndim_ > 0);
                this->contig_ = false;

                uint_fast64_t stride = 1;
                
                for(uint_fast8_t dim = this->ndim_; dim-- > 0;) {
                    this->strides_[dim] = stride;
                    stride *= shape[dim];
                    this->dim_sizes_[dim] = stride;
                    this->size_ *= shape[dim];
                }

                if(this->size_ != prev_size) {
                    throw std::invalid_argument("invalid shape - number of elements do not match");
                }
            }

            virtual void resize(const Shape &shape) {
                this->ndim_ = shape.size();
                this->shape_ = shape;
                this->strides_.resize(this->ndim_);
                this->dim_sizes_.resize(this->ndim_);
                this->size_ = this->ndim_ > 0;
                this->contig_ = true;

                uint_fast64_t stride = 1;
                
                for(uint_fast8_t dim = this->ndim_; dim-- > 0;) {
                    this->strides_[dim] = stride;
                    stride *= this->shape_[dim];
                    this->dim_sizes_[dim] = stride;
                    this->size_ *= this->shape_[dim];
                }
            }

            uint_fast64_t physical_ravel_ndindex(const NDIndex &ndindex) const noexcept {
                uint_fast64_t index = 0;

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    index += ndindex[dim] * this->strides_[dim];
                }

                return index;
            }

            uint_fast64_t logical_ravel_ndindex(const NDIndex &ndindex) const noexcept {
                uint_fast64_t index = 0;
                uint_fast64_t contig_stride = this->size_;

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    contig_stride /= this->shape_[dim];
                    index += ndindex[dim] * contig_stride;
                }

                return index;
            }

            NDIndex physical_unravel_index(uint_fast64_t index) const noexcept {
                NDIndex ndindex(this->ndim_);

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    ndindex[dim] = index / this->strides_[dim];
                    index %= this->strides_[dim];
                }

                return ndindex;
            }
            
            NDIndex logical_unravel_index(uint_fast64_t index) const noexcept {
                NDIndex ndindex(this->ndim_);
                uint_fast64_t contig_stride = this->size_;

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    contig_stride /= this->shape_[dim];
                    ndindex[dim] = index / contig_stride;
                    index %= contig_stride;
                }

                return ndindex;
            }

            void squeeze() noexcept {
                uint_fast8_t new_ndim = 0;

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    if(this->shape_[dim] > 1) {
                        this->shape_[new_ndim] = this->shape_[dim];
                        this->strides_[new_ndim] = this->strides_[dim];
                        this->dim_sizes_[new_ndim] = this->dim_sizes_[dim];
                        new_ndim++;
                    }
                }

                this->ndim_ = new_ndim;
                this->shape_.resize(this->ndim_);
                this->strides_.resize(this->ndim_);
                this->dim_sizes_.resize(this->ndim_);
            }

            std::string str() const noexcept {
                std::string str("shape = (");
                uint_fast8_t dim = 0;

                for(dim = 0;dim < this->ndim_ - 1;dim++) {
                    str += std::to_string(this->shape_[dim]);
                    str.push_back(',');
                    str.push_back(' ');
                }
                str += std::to_string(this->shape_[dim]) + ")\nstrides = (";

                for(dim = 0;dim < this->ndim_ - 1;dim++) {
                    str += std::to_string(this->strides_[dim]);
                    str.push_back(',');
                    str.push_back(' ');
                }

                str += std::to_string(this->strides_[dim]) + ")\ndim_sizes = (";

                for(dim = 0;dim < this->ndim_ - 1;dim++) {
                    str += std::to_string(this->dim_sizes_[dim]);
                    str.push_back(',');
                    str.push_back(' ');
                }
                
                str += std::to_string(this->dim_sizes_[dim]) + ")\nsize = " + 
                std::to_string(this->size_) + "\nndim = " +
                std::to_string(this->ndim_) + "\ncontiguous = " + std::to_string(this->contig_);
                str.push_back('\n');

                return str;
            }

            uint_fast64_t physical_index(uint_fast64_t logical_index) const noexcept {
                if(this->contig_) {
                    return logical_index;
                }

                uint_fast64_t cstride = this->size_;
                uint_fast64_t physical_index = 0;
                uint_fast64_t dim_index;

                for(uint_fast8_t dim = 0;dim < this->ndim_;dim++) {
                    cstride /= this->shape_[dim];
                    dim_index = logical_index / cstride;
                    logical_index %= cstride;
                    physical_index += this->strides_[dim] * dim_index;
                }

                return physical_index;
            }

            inline uint_fast64_t axes_size(const Axes &axes) const noexcept {
                uint_fast64_t size = axes.size() > 0;

                for(auto iter = axes.begin();iter != axes.end();iter++) {
                    size *= this->shape_[*iter];
                }

                return size;
            }

            inline Shape& shape() const noexcept {
                return this->shape_;
            }

            inline Strides& strides() const noexcept {
                return this->strides_;
            }

            inline Strides& dim_sizes() const noexcept {
                return this->dim_sizes_;
            }

            inline uint_fast64_t& size() const noexcept {
                return this->size_;
            }

            inline uint_fast8_t& ndim() const noexcept {
                return this->ndim_;
            }

            inline bool& contig() const noexcept {
                return this->contig_;
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const ArrayBase &base) noexcept {
                return stream << base.str();
            }
    };
}

#endif