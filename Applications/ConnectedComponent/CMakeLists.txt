# Top level directory has the include files
add_executable(lacc CC.cpp)
target_link_libraries(lacc CombBLAS)
