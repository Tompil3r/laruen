
#ifndef NDLIB_RNG_H_
#define NDLIB_RNG_H_

#include <random>

namespace laruen::ndlib {

    std::mt19937 RNG(std::random_device{}());
    
}


#endif