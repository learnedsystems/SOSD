#include "searches/linear_search.h"
#include "searches/branching_binary_search.h"
#include "searches/branchless_binary_search.h"
#include "searches/interpolation_search.h"


#define INSTANTIATE_TEMPLATES(func_name, type_name) template void func_name<LinearSearch>(sosd::Benchmark<type_name, LinearSearch>&, bool); template void func_name<BranchingBinarySearch>(sosd::Benchmark<type_name, BranchingBinarySearch>&, bool); template void func_name<BranchlessBinarySearch>(sosd::Benchmark<type_name, BranchlessBinarySearch>&, bool); template void func_name<InterpolationSearch>(sosd::Benchmark<type_name, InterpolationSearch>&, bool)

#define INSTANTIATE_TEMPLATES_RMI(func_name, type_name) template void func_name<LinearSearch>(sosd::Benchmark<type_name, LinearSearch>&, bool, const std::string&); template void func_name<BranchingBinarySearch>(sosd::Benchmark<type_name, BranchingBinarySearch>&, bool, const std::string&); template void func_name<BranchlessBinarySearch>(sosd::Benchmark<type_name, BranchlessBinarySearch>&, bool, const std::string&); template void func_name<InterpolationSearch>(sosd::Benchmark<type_name, InterpolationSearch>&, bool, const std::string&)
