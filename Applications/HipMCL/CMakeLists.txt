# Top level directory has the include files

add_executable(mcl MCL.cpp)
target_link_libraries(mcl CombBLAS)
add_subdirectory(Incremental)