#pragma once

#include <type_traits>

#if __cplusplus < 201402L
namespace std {

template<bool B, class T = void>
using enable_if_t = typename enable_if<B,T>::type;

}
#endif