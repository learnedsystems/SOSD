
set(SOURCE_FILES ${SOURCE_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/bitset.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intrinsics.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intrin_avx2.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intrin_avx512.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intrin_sse.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intrin_x64.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp

    PARENT_SCOPE)
