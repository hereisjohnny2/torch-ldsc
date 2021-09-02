#ifndef ROCKIMAGERGB_HPP
#define ROCKIMAGERGB_HPP

#include <torch/data.h>
#include <string>
#include <vector>
struct RockData {
    torch::Tensor rgbInfo, labels;
};

class RockImageRGBDataset : public torch::data::Dataset<RockImageRGBDataset>
{
private:
    RockData data;

public:
    RockImageRGBDataset(const std::string &stringData)
        : data(loadDataFromFile(stringData)) {}

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    const RockData readDataFromString(const std::string &stringData);
    const RockData loadDataFromFile(const std::string &filePath);
};

#endif // !ROCKIMAGERGB_HPP