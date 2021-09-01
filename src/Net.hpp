#include <torch/torch.h>

struct Net : torch::nn::Module 
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Net() 
    {
        fc1 = register_module("fc1", torch::nn::Linear(3,5));
        fc2 = register_module("fc2", torch::nn::Linear(5,5));
        fc3 = register_module("fc3", torch::nn::Linear(5,2));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 3})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

};
