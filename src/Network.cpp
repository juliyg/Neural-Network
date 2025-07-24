#include <cstdlib>      
#include <random>
#include <ctime>       
#include <algorithm>    
#include <iostream>    
#include "Network.hpp"  

Network::Network(int batch, double learnRate, const std::vector<int>& sizes) 
    : sizes(sizes), 
    zl(sizes.size()), 
    al(sizes.size()), 
    wl(sizes.size()-1), 
    bl(sizes.size()-1), 
    learnRate(learnRate), 
    batch(batch) {

    // Seed the random number generator properly
    std::srand(std::time(0));
    Eigen::initParallel();  // If using Eigen in parallel
    
    for(int i = 0; i < wl.size(); ++i) {
        double scale = std::sqrt(2.0 / sizes[i]); 
        wl[i] = scale * Eigen::MatrixXd::Random(sizes[i+1], sizes[i]); 
        bl[i] = Eigen::VectorXd::Random(sizes[i+1]); 
    }
}

Eigen::VectorXd Network::dSigmoid(const Eigen::VectorXd& al) {
    return (al.array() * (1-al.array())).matrix(); 
}

Eigen::VectorXd Network::sigmoid(const Eigen::VectorXd& zl) {
    return zl.unaryExpr([](double x) {
        return 1.0/(1.0 + std::exp(-x));
    });
}

double Network::cross_entropy(const Eigen::VectorXd& output, const Eigen::VectorXd& desired) {
    const double epsilon = 1e-12; 
    Eigen::ArrayXd a = output.array().max(epsilon).min(1-epsilon); 
    double loss = -(
        desired.array() * a.log() + 
        (1-desired.array()) * 
        (1.0-a).log()
    ).sum();
    return loss; 
}


void Network::feedforward(const Eigen::VectorXd& input) {
    al[0] = input; 

    for(int i = 0; i < wl.size(); ++i) {
        zl[i] = wl[i] * al[i] + bl[i]; 
        al[i + 1] = sigmoid(zl[i]); 
    }
}

void Network::backprop(
    const Eigen::VectorXd& desired,
    std::vector<Eigen::MatrixXd>& nabla_w,
    std::vector<Eigen::VectorXd>& nabla_b) {

    Eigen::VectorXd deltaL = al[sizes.size()-1] - desired; 
    
    for(int i = wl.size()-1; i >= 0; --i) {
        nabla_w[i] += deltaL * al[i].transpose();
        nabla_b[i] += deltaL; 
        if(i >= 1)
            deltaL = ((wl[i].transpose() * deltaL).array() * dSigmoid(al[i]).array()).matrix();
    }
}


void Network::gradientDescent(
    const std::vector<Eigen::VectorXd>& input,
    const std::vector<Eigen::VectorXd>& desired) {
    std::vector<Eigen::MatrixXd> nabla_w(wl.size());
    std::vector<Eigen::VectorXd> nabla_b(bl.size());

    for(int i = 0; i < nabla_w.size(); ++i) {
        nabla_w[i] = Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]); 
        nabla_b[i] = Eigen::VectorXd::Zero(sizes[i+1]); 
    }

    for(int i = 0; i < input.size(); ++i) {
        feedforward(input[i]);
        backprop(desired[i], nabla_w, nabla_b);
    }

    for(int i = 0; i < wl.size(); ++i) {
        wl[i] -= (learnRate/input.size() * nabla_w[i]);
        bl[i] -= (learnRate/input.size() * nabla_b[i]); 
    }
}

void Network::mini_batch_train(
        const std::vector<Eigen::VectorXd>& input,
        const std::vector<Eigen::VectorXd>& output) {
    
    int dataSize = input.size(); 

    std::vector<int> indices(dataSize); 
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen); 

    for(int i = 0; i < dataSize; i += batch) {
        std::vector<Eigen::VectorXd> miniInput;
        std::vector<Eigen::VectorXd> miniOutput;

        for(int j = 0; j < batch && i + j < dataSize; ++j) {
            miniInput.push_back(input[indices[i + j]]); 
            miniOutput.push_back(output[indices[i+j]]);
        }

        gradientDescent(miniInput, miniOutput);
    }
}

void Network::evaluate(
        const std::vector<Eigen::VectorXd>& input,
        const std::vector<Eigen::VectorXd>& output,
        const int verbosity) {

    int dataSize = input.size(); 
    double correct = 0;

    for(int i = 0; i < dataSize; ++i) {
        feedforward(input[i]); 
        int predicted_value;
        int actual_value;
        double maxPredict = al[sizes.size()-1].maxCoeff(&predicted_value);
        double maxActual = output[i].maxCoeff(&actual_value);
        
        if(predicted_value == actual_value) {
            correct++; 
            if(i % verbosity == 0) {
            std::cout << "Correct. Predicted: " 
                << predicted_value << ", Actual: " 
                << actual_value << ", Activation: " 
                << maxPredict << ", Percent correct: " 
                << correct/(i+1) << std::endl;
            }
        } else {
            if(i % verbosity == 0) {
            std::cout << "Incorrect. Predicted: " 
                << predicted_value << ", Actual: " 
                << actual_value << ", Activation: " 
                << maxPredict << ", Percent correct: " 
                << correct/(i+1) << std::endl;
            }
        }
    }

    std::cout << "\nFinal Accuracy: " 
              << static_cast<double>(correct) / dataSize * 100.0 
              << "%" << std::endl;
}

