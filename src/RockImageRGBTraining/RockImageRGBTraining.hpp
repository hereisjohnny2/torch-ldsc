#include <memory>
#include <iomanip>

#include <torch/torch.h>
#include "../RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "../RockImageRGBNet/RockImageRGBNet.hpp"

class RockImageRGBTraining
{
private:
    RockImageRGBDataset dataset;
    std::shared_ptr<RockImageRGBNet> net;

public:
    RockImageRGBTraining(RockImageRGBDataset _dataset, std::shared_ptr<RockImageRGBNet> _net) 
        : dataset(_dataset), net(_net) {};

    void execute(int batch_size, int n_epoch, float lr) {}
        
};