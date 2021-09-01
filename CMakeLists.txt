cmake_minimum_required(VERSION 3.16.0)

project(torch-ldsc VERSION 0.1.0)

# Add the pytorch library from a local location
list(APPEND CMAKE_PREFIX_PATH "/home/joao/Documentos/dev/C++/libtorch")
find_package(Torch REQUIRED)

include(CTest)
enable_testing()

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/RockImageRGBDataset.cpp
    src/RockImageRGBDataset.hpp
    src/Net.hpp
)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
