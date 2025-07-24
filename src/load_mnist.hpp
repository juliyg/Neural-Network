#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include "Eigen/Dense"

std::vector<Eigen::VectorXd> load_mnist_images(const char* filename); 
std::vector<Eigen::VectorXd> load_mnist_labels(const char* filename); 
