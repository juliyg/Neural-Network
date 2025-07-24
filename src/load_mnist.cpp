#include "load_mnist.hpp"
#include <stdexcept> 
#include <vector>

uint32_t extract(std::ifstream& f) {
    unsigned char b[4]; 
    f.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]); 
}

std::vector<Eigen::VectorXd> load_mnist_images(const char* filename) {
    std::ifstream file(filename, std::ios::binary); 
    if (!file) throw std::runtime_error("Could not open image file"); 

    uint32_t magic = extract(file); 
    if (magic != 2051) throw std::runtime_error("Invalid image file (magic number wrong)");

    uint32_t num_images = extract(file); 
    uint32_t rows = extract(file);
    uint32_t cols = extract(file); 

    std::vector<Eigen::VectorXd> images(num_images, Eigen::VectorXd::Zero(rows*cols)); 
    for(int i = 0; i < num_images; ++i) {
        for(int j = 0; j < rows*cols; ++j) {
            uint8_t raw;
            file.read(reinterpret_cast<char*>(&raw), 1);
            double norm = raw/255.0; 
            (images[i])[j] = norm; 
        }
    }

    return images; 
}


std::vector<Eigen::VectorXd> load_mnist_labels(const char* filename) {
    std::ifstream file(filename, std::ios::binary); 
    if (!file) throw std::runtime_error("Could not open label file"); 

    uint32_t magic = extract(file); 
    if(magic != 2049) throw std::runtime_error("Invalid label file (magic number wrong)"); 

    uint32_t label_count = extract(file); 

    std::vector<Eigen::VectorXd> labels(label_count, Eigen::VectorXd::Zero(10)); 
    for(int i = 0; i < label_count; ++i) {
        uint8_t raw;
        file.read(reinterpret_cast<char*>(&raw), 1); 
        (labels[i])[raw] = 1.0; 
    }   

    return labels; 
}