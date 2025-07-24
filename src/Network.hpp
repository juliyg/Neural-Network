#include "Eigen/Dense"
#include <vector>

class Network {
private: 
    // Number of neurons per layer 
    std::vector<int> sizes; 

    // Input function per layer 
    std::vector<Eigen::VectorXd> zl; 

    //Activation function per layer 
    std::vector<Eigen::VectorXd> al; 

    // Network weights and biases 
    // Network weights per layer 
    std::vector<Eigen::MatrixXd> wl; 

    //Network biases per layer 
    std::vector<Eigen::VectorXd> bl; 

    //Learning Rate 
    double learnRate;

    //Batch size
    int batch; 

public: 
    // Constructor for the network
    // Arguments
    // - batch: batch size
    // - learnRate: learning rate 
    // - sizes: sizes for each layer 
    Network(int batch, double learnRate, const std::vector<int>& sizes);

    // Computes gradients of the cost function via backpropagation
    // Arguments:
    // - desired: true label (desired output of the network)
    // - nabla_w: output gradient of weights per layer
    // - nabla_b: output gradient of biases per layer
    void backprop(
    const Eigen::VectorXd& desired,
    std::vector<Eigen::MatrixXd>& nabla_w,
    std::vector<Eigen::VectorXd>& nabla_b); 


    
    // Performs a forward pass and stores all activations and z-values
    // Argument:
    // - input: input vector to the network (e.g., input layer)
    void feedforward(const Eigen::VectorXd& input);

    
    // Calculates activation using sigmoid function 
    // Argument:
    // - zl: Z^L
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& zl); 

    // Calculates dA/dZ 
    // Arguments
    // -al: a^l 
    Eigen::VectorXd dSigmoid(const Eigen::VectorXd& al); 


    // Cross-Entropy Loss 
    // Arugments 
    // - output: output of NN
    // - desired: desired output of NN 
    double cross_entropy(const Eigen::VectorXd& output, const Eigen::VectorXd& desired); 

    // Update gradient for one batch  
    // Arguments 
    // - learnRate: learning rate for gradient descent 
    void gradientDescent(
        const std::vector<Eigen::VectorXd>& input,
        const std::vector<Eigen::VectorXd>& desired); 

    // Collect batches and create inputs 
    // Arguments 
    // - input: all possible inputs to the network parsed from database
    // - output: corresponding label data to the image
    void mini_batch_train(
        const std::vector<Eigen::VectorXd>& input,
        const std::vector<Eigen::VectorXd>& output
    );

    // Evaulates the neural network using the dataset 
    // Arguments 
    // - input: all possible inputs to the network parsed from database
    // - output: corresponding label data to the image
    // - verbosity: number of iterations per output
    void evaluate(
        const std::vector<Eigen::VectorXd>& input,
        const std::vector<Eigen::VectorXd>& output,
        const int verbosity
    );
}; 