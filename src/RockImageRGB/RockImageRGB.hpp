#ifndef ROCKIMAGERGB_HPP
#define ROCKIMAGERGB_HPP

#include <string>
#include <torch/torch.h>
#include "./RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "./RockImageRGBNet/RockImageRGBNet.hpp"

struct RGBValueDTO
{
    torch::Tensor rgb;

    RGBValueDTO(double _rgb[])
    {
        rgb = torch::from_blob(_rgb, {1, 3});
    }
};

class RockImageRGB
{
private:
    std::shared_ptr<RockImageRGBNet> model;

public:
    RockImageRGB(
        std::shared_ptr<RockImageRGBNet> _model
    ) : model(_model) {}

    void train(
        std::string train_dataset, 
        torch::optim::Optimizer &optimizer, 
        int batch_size = 5,
        int num_epochs = 1000
    );
    void test(std::string test_dataset) {}
    int runModel(const RGBValueDTO &rgbValue);
};





#endif // !ROCKIMAGERGB_HPP