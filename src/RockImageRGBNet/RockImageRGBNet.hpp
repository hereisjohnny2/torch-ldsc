#ifndef ROCKIMAGERGBNET_HPP
#define ROCKIMAGERGBNET_HPP

#include <torch/torch.h>
struct RockImageRGBNet : torch::nn::Module 
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    RockImageRGBNet();
    torch::Tensor forward(torch::Tensor x);
};

#endif // !ROCKIMAGERGBNET_HPP
