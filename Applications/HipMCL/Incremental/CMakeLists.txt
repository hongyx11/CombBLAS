# Top level directory has the include files

## Add timing flag for timing breakdown prints
## https://stackoverflow.com/questions/44284275/passing-compiler-options-in-cmake
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTIMING")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMING")

add_executable(inc INC.cpp)
target_link_libraries( inc CombBLAS )

# add option to hide the debug build of incremental project
option(INC_DETAILED "Incremental project debug build" OFF)
if(INC_DETAILED)
add_executable(full Full.cpp)
add_executable(testideas Test.cpp)
add_executable(prep-data Prep-Data.cpp)
add_executable(prep-data-metaclust Prep-Data-Metaclust.cpp)
add_executable(inc-pipeline Incremental-Pipeline.cpp)
add_executable(inc-baseline-pipeline Incremental-Baseline-Pipeline.cpp)
add_executable(inc-toy-pipeline Incremental-Toy-Pipeline.cpp)
target_link_libraries( full CombBLAS )
target_link_libraries( testideas CombBLAS )
target_link_libraries( prep-data CombBLAS )
target_link_libraries( prep-data-metaclust CombBLAS )
target_link_libraries( inc-pipeline CombBLAS )
target_link_libraries( inc-baseline-pipeline CombBLAS )
target_link_libraries( inc-toy-pipeline CombBLAS )
endif(INC_DETAILED)