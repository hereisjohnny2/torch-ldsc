#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include "utils.hpp"

std::string utils::readDataFromFile(const std::string &filename) {
    std::ifstream file(filename);    
    file.open(filename);    

    if (!file.is_open()) {
        throw std::invalid_argument("Coundn't find the file!");  
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}