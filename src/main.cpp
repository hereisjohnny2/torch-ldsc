#include <iostream>
#include <string>
#include <iomanip>
#include <memory>

#include <torch/optim.h>

#include "RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "RockImageRGBNet/RockImageRGBNet.hpp"
#include "utils/utils.hpp"

template<typename DataLoader>
void executeTraining(
    int epoch,
    std::shared_ptr<RockImageRGBNet> model,
    DataLoader& data_loader,
    torch::optim::Optimizer &optimizer,
    int datasetSize
) {
    model->train();
    size_t batchIndex = 0;
    
    for(auto &batch : data_loader)
    {
        auto data = batch.data;
        auto target = batch.target.squeeze();

        data = data.to(torch::kF32);
        target = target.to(torch::kInt64);

        optimizer.zero_grad();

        auto output = model->forward(data);
        auto loss = torch::nll_loss(output, target);

        loss.backward();
        optimizer.step();

        if (batchIndex++ % 10 == 0) {
            std::printf(
                "\rEpoch: %d [%5ld/%5d] Loss: %f",
                epoch,
                batchIndex * batch.data.size(0),
                datasetSize,
                loss.template item<float>()
            );
        }
        
    }
}

int main(int argc, const char** argv) {
    double lr = 0.04;
    int batch_size = 5;

    if (argc == 3) {
        lr = std::atof(argv[1]);
        batch_size = std::atoi(argv[2]);
    }

    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string trainingData = utils::readDataFromFile(filename);

    auto dataset = RockImageRGBDataset(trainingData).map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        /*batch_size = */batch_size
    );

    auto net = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(net->parameters(), /*lr = */lr);

    auto datasetSize = dataset.size().value();
    for (int epoch = 0; epoch <= 1000; epoch++)
    {
        executeTraining(
            epoch,
            net,
            *dataLoader,
            optimizer,
            datasetSize
        );
        // for (auto &batch : *dataLoader)
        // {
        //     auto data = batch.data;
        //     auto target = batch.target.squeeze();

        //     data = data.to(torch::kF32);
        //     target = target.to(torch::kInt64);

        //     optimizer.zero_grad();

        //     auto output = net->forward(data);
        //     auto loss = torch::nll_loss(output, target);

        //     loss.backward();
        //     optimizer.step();

        //     std::cout 
        //         << "Train Epoch: "
        //         << epoch
        //         << " Loss: "
        //         << loss.item<float>()
        //         << std::endl;
            
        // }
    }

    torch::save(net, "/home/joao/Documentos/dev/C++/test-pytorch/data/model.dat");    

    return 0;
}