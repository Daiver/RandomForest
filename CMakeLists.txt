cmake_minimum_required(VERSION 2.8) 
SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++-3.5")
SET (CMAKE_CXX_FLAGS "-Wall -Wno-sign-compare  -O3 --std=c++11  -Wunused-parameter -Wstrict-overflow=5 -Wundef -Wcast-qual -Wcast-align -Wmissing-declarations -Wredundant-decls -Wdisabled-optimization -Woverloaded-virtual -Wctor-dtor-privacy -Wno-reorder")


include_directories(include)

set(SOURCE_LIB src/common.cpp src/Node.cpp src/DecisionTree.cpp src/RandomForest.cpp)
add_library(RandomForest STATIC ${SOURCE_LIB})

add_executable(main main.cpp)
target_link_libraries(main RandomForest)
