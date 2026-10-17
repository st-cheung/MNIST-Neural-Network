# MNIST-Neural-Network
Handwritten Digit Classification with Neural Networks

## Overview
A MATLAB implementation of a feed-forward neural network for handwritten digit classification using the MNIST dataset.

This project was completed as my undergraduate final-year project, with the implementation developed by me under academic supervision. I investigated how network design and training choices affected classification performance.

## Approach
- Implemented a feed-forward neural network with backpropagation and stochastic gradient descent.

- Experimented with network architecture, training parameters and regularisation.

- Evaluated model performance on unseen data and analysed experimental results.

## Results
- The final model achieved approximately 98% accuracy on the MNIST test set.

### Effect of Regularisation

The training curves compare the performance of the unregularised and regularised networks, showing how regularisation affected training behaviour.

### Confusion Matrix

The confusion matrix shows the classification performance of the final model across the ten digit classes.

## Files
- neural_net.m - main neural network implementation and training procedure.
- MNIST_classifier.m - loads the MNIST dataset and runs the classification experiment.
- load_MNIST.m - converts the MNIST data into vectorised inputs suitable for the neural network.
- forward_prop.m - performs forward propagation through the network.
- backprop.m - performs backpropagation through the network.
- cost_function.m - defines the objective used to evaluate the network's predictions.
- leaky_relu.m - implements the Leaky ReLU activation function.
