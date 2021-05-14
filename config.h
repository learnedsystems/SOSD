#ifndef SOSD_CONFIG_H
#define SOSD_CONFIG_H

//#define USE_FAST_MODE

namespace sosd_config {
#ifdef USE_FAST_MODE
constexpr bool fast_mode = true;
#else
constexpr bool fast_mode = false;
#endif
}  // namespace sosd_config

#endif
