add_library(RockImageRGBNet
    RockImageRGBNet.cpp
)

target_link_libraries(RockImageRGBNet
    "${TORCH_LIBRARIES}"
)