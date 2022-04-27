
#ifndef NDLIB_H
#define NDLIB_H

#include "src/ndlib/ndarray_types.h"
#include <cstdint>

namespace laruen::ndlib {
   /*
      dir_broadcast - directional broadcast - shape1.size() >= shape2.size() (shape1 has the same
      or more dimensions as shape2)
      broadcast - no assumptions about the shapes is made (uses dir_broadcast)
   */
   Shape dir_broadcast(const Shape &shape1, const Shape &shape2);
   Shape broadcast(const Shape &shape1, const Shape &shape2);
};

#include "src/ndlib/ndlib.tpp"
#endif