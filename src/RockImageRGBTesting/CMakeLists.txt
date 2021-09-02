add_library(RockImageRGBTesting
    RockImageRGBTesting.cpp
)

target_link_libraries(RockImageRGBTesting
    "${TORCH_LIBRARIES}"
)