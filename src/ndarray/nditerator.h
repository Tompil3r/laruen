
#ifndef NDITERATOR_H
#define NDITERATOR_H

namespace laruen::ndarray {

    template <typename T>
    class NDIterator {
        T &ndarray;
        uint64_t index;
        NDIndex ndindex;

        public:
            NDIterator() = delete;
            NDIterator(T &ndarray);

            auto& next();
            void reset();

            inline bool has_next() {
                return this->ndindex[0] < this->ndarray.shape[0];
            }
    };
};

#include "src/ndarray/nditerator.tpp"
#endif