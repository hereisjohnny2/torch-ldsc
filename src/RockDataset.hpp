#include <torch/torch.h>
#include <string>
#include <vector>

struct RockData {
    torch::Tensor rgbInfo, labels;
};

class RockDataset : public torch::data::Dataset<RockDataset>
{
private:
    RockData data;

public:
    RockDataset(const std::string &stringData)
        : data(readDataFromString(stringData)) {}

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    const RockData readDataFromString(const std::string &stringData);
};
