
#ifndef MULTI_RNG_H_
#define MULTI_RNG_H_

#include <random>

namespace laruen::multi {

    std::mt19937 RNG(std::random_device{}());
    
}


#endif