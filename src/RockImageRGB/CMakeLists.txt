add_subdirectory(RockImageRGBDataset)
add_subdirectory(RockImageRGBNet)
add_subdirectory(RockImageRGBTraining)
add_subdirectory(RockImageRGBTesting)

add_library(RockImageRGB RockImageRGB.cpp)

target_link_libraries(RockImageRGB
    "${TORCH_LIBRARIES}" 
    RockImageRGBDataset
    RockImageRGBNet
    RockImageRGBTraining
    RockImageRGBTesting
)