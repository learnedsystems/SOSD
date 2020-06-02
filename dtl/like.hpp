#pragma once

#include <regex>
#include <string>

#include <dtl/dtl.hpp>


namespace dtl {


struct like {

  const std::regex pattern;

  like(const std::string& like_pattern)
      : pattern(std::regex(std::regex_replace(std::regex_replace(like_pattern, std::regex("%"), ".*"), std::regex("_"), "."))) { };

  inline u1
  match(const std::string& input) {
    return std::regex_match(input, pattern);
  }

  inline u1
  operator()(const std::string& input) {
    return match(input);
  }

};


struct ilike {

  const std::regex pattern;

  ilike(const std::string& like_pattern)
      : pattern(std::regex(std::regex_replace(std::regex_replace(like_pattern, std::regex("%"), ".*"), std::regex("_"), "."), std::regex_constants::icase)) { };

  inline u1
  match(const std::string& input) {
    return std::regex_match(input, pattern);
  }

  inline u1
  operator()(const std::string& input) {
    return match(input);
  }

};


} // namespace dtl