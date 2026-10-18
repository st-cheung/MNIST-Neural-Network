# Handwritten Digit Classification with Neural Networks

## Overview
A MATLAB implementation of a feed-forward neural network for handwritten digit classification using the MNIST dataset.

This project was completed as my undergraduate final-year project. I developed the implementation and investigated how network design and training choices affected classification performance. 

## Approach
Implemented a feed-forward neural network with backpropagation and stochastic gradient descent.

Experimented with activation functions, training parameters and regularisation, with a particular focus on the effect of regularisation on generalisation.

Evaluated model performance on unseen data and analysed experimental results.

## Results
The final model achieved approximately 98% accuracy on the MNIST test set.

### Effect of Regularisation
The training curves compare the performance of the unregularised (left) and regularised (right) networks, showing how regularisation affected training behaviour.
![Error rates for the unregularised (left) and regularised (right) networks over the training (blue) and validation (red) sets](/results/regularisation_error.png)

### Confusion Matrix
The confusion matrix shows the classification performance of the final model across the ten digit classes.
![Confusion matrix for the final model](/results/confusion_matrix.png)

## Files
The repository contains MATLAB scripts and functions for:<br/>
`neural_net.m` - main neural network implementation and training procedure. <br/>
`MNIST_classifier.m` - loads the MNIST dataset and runs the classification experiments.<br/>
`load_MNIST.m` - converts the MNIST data into vectorised inputs suitable for the neural network.<br/>
`forward_prop.m` - performs forward propagation through the network.<br/>
`backprop.m` - performs backpropagation through the network.<br/>
`cost_function.m` - defines the objective function used to evaluate the network's predictions.<br/>
`leaky_relu.m` - implements the Leaky ReLU activation function.<br/>
`learning_rate.m` - performs the learning rate adjustments.<br/>
