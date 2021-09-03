add_library(RockImageRGBTraining
    RockImageRGBTraining.cpp
)

target_link_libraries(RockImageRGBTraining
    "${TORCH_LIBRARIES}"
)