#pragma once

class Competitor {
 public:
  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  uint64_t additional_build_time() const {
    return 0;
  }
};
