
#ifndef STRINGS_H
#define STRINGS_H

#include <string>
#include <sstream>
#include <type_traits>

namespace laruen::utils::strings {
    
    template <typename T>
    std::string to_string(T number) noexcept {
        if constexpr(std::is_floating_point_v<T>) {
            std::stringstream sstrm;

            if constexpr(sizeof(T) <= 4) {
                sstrm.precision(7);
            }
            else {
                sstrm.precision(15);
            }
            sstrm << number;
            return sstrm.str();
        }
        else {
            return std::to_string(number);
        }
    }
}

#endif