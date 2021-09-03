#include <memory>
#include <iomanip>

#include <torch/optim.h>

#include "../RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "../RockImageRGBNet/RockImageRGBNet.hpp"

class RockImageRGBTesting
{
private:
    std::shared_ptr<RockImageRGBNet> model;
    torch::optim::Optimizer &optimizer;

public:
    RockImageRGBTesting(
        std::shared_ptr<RockImageRGBNet> _model,
        torch::optim::Optimizer &_optimizer
    ) : model(_model), optimizer(_optimizer) {};

    template<typename DataLoader>
    void execute(int epoch, int datasetSize, DataLoader &dataLoader) 
    {
        torch::NoGradGuard();
        model->eval();

        double testLoss = 0;
        int correct = 0;
        
        for(auto &batch : dataLoader)
        {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);

            auto output = model->forward(data);
            testLoss += torch::nll_loss(output, target, {}, torch::Reduction::Sum).template item<double>();

            auto pred = output.argmax(1);
            correct += pred.eq(target).sum().template item<int64_t>();
        }

        testLoss /= datasetSize;
        std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            testLoss,
            static_cast<double>(correct) / datasetSize
        );
    }    
};
