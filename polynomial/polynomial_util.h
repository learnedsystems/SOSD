#ifndef SOSDB_POLYNOMIAL_UTIL_H
#define SOSDB_POLYNOMIAL_UTIL_H

#include <cmath>

namespace polynomial_spline {
  // For polynomial search
  using segCoord = std::pair<double, uint32_t>;
  using Coord = std::pair<double, double>;
  using Errors = std::pair<double, double>; // could also be changed to support relative error

  // Used by Remez's algorithm
  std::vector<segCoord> globalSpline;

// This part of code doesn't have to be performant regarding execution time
// It's used in the Build method
namespace polynomial_util {
  // for comparisons between floating points
  static constexpr double precision = std::numeric_limits<double>::epsilon();
  
  static double hornerEvaluation(std::vector<double>& currPoly, uint64_t x)
  // currPoly comes from the Remez's algorithm
  // evaluate the polynomial with Horner's schema
  {
    uint32_t currPolyDegree = currPoly.size() - 1;
    double ret = currPoly[currPolyDegree];
    for (unsigned i = currPolyDegree; i != 0; --i)
        ret = ret * x + currPoly[i - 1];
    return static_cast<double>(ret);
  }
    
  static std::pair<double, double> getPolyFitErrors(std::vector<segCoord>& segmentSplineToFit, std::vector<double> currPoly)
  // compute the errors of currPoly applied to the segment spline
  {
    std::pair<double, double> errs = std::make_pair(0, 0);
    for (auto elem: segmentSplineToFit) {
      double eval = hornerEvaluation(currPoly, elem.first);
      double error = fabs(eval - elem.second);
      
      // Bigger error found?
      if (error > errs.second)
        errs.second = error;
      errs.first += error;
    }
    errs.first /= segmentSplineToFit.size();
    return errs;
  }
  
  static void printErrors(std::vector<segCoord>& segmentSplineToFit, std::vector<double> currPoly) __attribute__((unused));
  static void printErrors(std::vector<segCoord>& segmentSplineToFit, std::vector<double> currPoly)
  // debug errors
  {
    std::pair<double, double> errs = getPolyFitErrors(segmentSplineToFit, currPoly);
    std::cout << "spline_size: " << segmentSplineToFit.size() << std::endl << "Errors for polynomial (average, max): (" << errs.first << ", " << errs.second << ")" << std::endl;
  }
  
  static constexpr double golden_ratio = 1.61; // use it for the error spline
  
};

}

#endif //SOSDB_POLYNOMIAL_UTIL_H
  
