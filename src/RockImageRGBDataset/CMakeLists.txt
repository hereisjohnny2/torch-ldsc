add_library(RockImageRGBDataset
    RockImageRGBDataset.cpp
)

target_link_libraries(RockImageRGBDataset
    "${TORCH_LIBRARIES}"
)