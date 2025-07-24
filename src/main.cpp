#include <cstdlib>
#include <vector>
#include "Network.hpp"
#include <iostream>
#include "load_mnist.hpp"

int main(int argc, char* argv[]) {

    const char* imagesTrain = argv[1]; 
    const char* labelsTrain = argv[2]; 
    const char* imagesTest = argv[3]; 
    const char* labelsTest = argv[4]; 


    std::vector<Eigen::VectorXd> imagesTr = load_mnist_images(imagesTrain); 
    std::vector<Eigen::VectorXd> labelsTr = load_mnist_labels(labelsTrain); 
    std::vector<Eigen::VectorXd> imagesTe = load_mnist_images(imagesTest); 
    std::vector<Eigen::VectorXd> labelsTe = load_mnist_labels(labelsTest); 
    std::vector<int> sizes = {784, 100, 10}; 

    Network network(30, 0.04, sizes);

    for (int epoch = 0; epoch < 50; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << std::endl;
        network.mini_batch_train(imagesTr, labelsTr);
        network.evaluate(imagesTe, labelsTe, 1000);
    }
    return 0; 
}