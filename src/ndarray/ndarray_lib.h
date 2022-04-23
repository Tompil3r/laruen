
#ifndef NDARRAY_LIB_H
#define NDARRAY_LIB_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>

namespace laruen::ndarray {
   /*
      d_broadcast - directional broadcast - shape1.size() >= shape2.size() (shape1 has the same
      or more dimensions as shape2)
      broadcast - no assumptions about the shapes is made (uses d_broadcast)
   */
   Shape d_broadcast(const Shape &shape1, const Shape &shape2);
   Shape broadcast(const Shape &shape1, const Shape &shape2);
   bool equal_dims(const Shape &shape1, const Shape &shape2);
};

#include "src/ndarray/ndarray_lib.tpp"
#endif