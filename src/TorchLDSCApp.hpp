#ifndef TORCHLDSCAPP_HPP
#define TORCHLDSCAPP_HPP

#include <string>
#include <memory>
#include "RockImageRGB/RockImageRGB.hpp"


class TorchLDSCApp
{
public:
    void run();

private:
    void runTraining();
    void runModel();
    void showMainMenu();

    void saveModel(std::shared_ptr<RockImageRGBNet> model, const std::string &filename);
    void loadModel(std::shared_ptr<RockImageRGBNet> model, const std::string &filename);
    
};

#endif // TORCHLDSCAPP_HPP
