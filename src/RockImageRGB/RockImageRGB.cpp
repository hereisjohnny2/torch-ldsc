#include <torch/data.h>

#include "./RockImageRGB.hpp"
#include "./RockImageRGBTraining/RockImageRGBTraining.hpp"

void RockImageRGB::train(
    std::string train_dataset, 
    torch::optim::Optimizer &optimizer,
    int batch_size,
    int num_epochs
) 
{
    auto dataset = RockImageRGBDataset(train_dataset).map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        batch_size
    );
    auto datasetSize = dataset.size().value();

    RockImageRGBTraining train(model, optimizer);

    for (int epoch = 0; epoch <= num_epochs; epoch++)
    {
        train.execute(epoch, datasetSize, *dataLoader);
    }
}


int RockImageRGB::runModel(const RGBValueDTO &rgbValue) 
{ 
    auto output = model->forward(rgbValue.rgb);
    auto prediction = output.argmax(1).abs();

    return prediction.template item<int>();
}