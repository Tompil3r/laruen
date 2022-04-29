
#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <cstdint>

namespace laruen::utils::containers {

    template <typename T1, typename T2>
    inline void copy(T1 &lhs, T2 &rhs, uint64_t amount, uint64_t lhs_idx = 0, uint64_t rhs_idx = 0) {
        uint64_t lhs_max = lhs_idx + amount;

        for(;lhs_idx < lhs_max;lhs_idx++) {
            lhs[lhs_idx] = rhs[rhs_idx];
            rhs_idx++;            
        }
    }
    
    template <typename T1, typename T2>
    inline void copy(T1 &lhs, T2 &rhs, uint64_t lhs_idx = 0, uint64_t rhs_idx = 0) {
        uint64_t lhs_max = lhs_idx + lhs.size();

        for(;lhs_idx < lhs_max;lhs_idx++) {
            lhs[lhs_idx] = rhs[rhs_idx];
            rhs_idx++;            
        }
    }
}

#endif