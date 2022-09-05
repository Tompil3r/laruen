
#ifndef LARUEN_MULTI_RNG_H_
#define LARUEN_MULTI_RNG_H_

#include <random>

namespace laruen::multi {

    std::mt19937 RNG(std::random_device{}());
    
}


#endif