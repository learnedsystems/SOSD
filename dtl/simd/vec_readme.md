# SIMD Vectorization

The header `<dtl/simd.hpp>` provides an abstraction layer for 
x86 SIMD intrinsics.

It offers vectorized primitives which compile on any platform. 

At compile time the vector operations are translated to the 
corresponding ISA of the underlying hardware.
A compatibility layer is available if SIMD is not supported 
on the target platform.

# Glossary

- **masked operation**, **zero-masking**, **merge-masking**: //TBD 
- 


## Common functions signatures

## Special function signatures

- `blend(m, a, b)`: is by definition a masked operation. Thus, it 
  requires a `mask` but no `src` as in merge-masking operations.  

- `gather(...)`: requires a *base address*, a *scale* factor and an *index 
  vector*. In general the type of the index vector is not the same as
  the type of the gathered elements. Especially, if 32-bit elements are 
  gathered from 64-bit addresses, the function is considered as 
  *type-narrowing*.

# Known Issues
- gather/scatter from absolute addresses works works only for 64-bit types.
  It is not possible to gather values of other types, such as i32 etc.
  