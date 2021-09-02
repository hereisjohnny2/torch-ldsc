#include <memory>
#include <iomanip>

#include <torch/torch.h>
#include "../RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "../RockImageRGBNet/RockImageRGBNet.hpp"

class RockImageRGBTraining
{
private:
    std::shared_ptr<RockImageRGBNet> model;
    torch::optim::Optimizer &optimizer;

public:
    RockImageRGBTraining(
        std::shared_ptr<RockImageRGBNet> _model,
        torch::optim::Optimizer &_optimizer
    ) : model(_model), optimizer(_optimizer) {};

    template<typename DataLoader>
    void execute(int epoch, int datasetSize, DataLoader &dataLoader) 
    {
        model->train();
        size_t batchIndex = 0;
        
        for(auto &batch : dataLoader)
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
        
};