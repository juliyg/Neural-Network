# About

This project implements a fully configurable feedforward neural network from scratch in C++, using the Eigen linear algebra library. It supports training on the MNIST dataset using mini-batch gradient descent, sigmoid activations, and cross-entropy loss. The network is trained directly on raw `.ubyte` image and label files, parsed via custom byte-level I/O.

I built this project to understand how neural networks learn, and to explore the algorithms that drive deep learning from first principles.

# References

I closely followed the mathematical treatment and intuition from Michael Nielsen’s book, *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)*, which was an invaluable resource for understanding where each equation comes from and how they connect within the learning algorithm.

# Dependencies 

This project requires: 

- *[Eigen](https://eigen.tuxfamily.org/)* – C++ template library for linear algebra

To get started, clone this repository and make sure Eigen is available in your include path. You can either install it system-wide or clone the headers into a local directory.

# Dataset 

You'll need the MNIST dataset of handwritten numbers for this project, *[MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data)*

# Customization

The training setup is configured inside `main.cpp`:

```cpp
int batchSize = 30;
double learningRate = 0.04;
int epochs = 50;
std::vector<int> layers = {784, 100, 10};
```

# Usage

Once you’ve built the project, you can train and evaluate the neural network using:

```bash
./neural_net train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images.idx3-ubyte t10k-labels.idx1-ubyte (See main.cpp for more details)
```





