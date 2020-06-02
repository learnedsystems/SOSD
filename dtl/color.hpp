#pragma once

#include <ostream>

namespace dtl {

enum class color : uint32_t {
  red          = 31,
  green        = 32,
  light_green  = 92,
  yellow       = 33,
  light_yellow = 93,

  blue         = 34,
  light_blue   = 94,
  dark_gray    = 90,
  gray         = 39,
  BG_RED       = 41,
  BG_GREEN     = 42,
  BG_BLUE      = 44,
  bg_gray      = 49,
};

struct color_modifier {
  color code;

  color_modifier(color code) : code(code) {}
  friend std::ostream&
  operator<<(std::ostream& os, const color_modifier& mod) {
    return os << "\033[" << static_cast<uint32_t>(mod.code) << "m";
  }
};

} // namespace dtl