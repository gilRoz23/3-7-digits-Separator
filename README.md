# Soft-SVM Implementation for MNIST Dataset

This program implements a Soft-SVM (Support Vector Machine) algorithm for the MNIST dataset. It includes functions to train the SVM model using different experiments and to visualize the results.

## Prerequisites

- Python 3.x
- numpy
- cvxopt
- matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git

2. Install the required dependencies:
   ```bash
   pip install numpy cvxopt matplotlib

## Usage
1. Ensure you have the EX2q2_mnist.npz file containing the MNIST dataset in the same directory as the script.
2. Run the script:
  ```bash
  python softsvm_mnist.py
  ```
3. The program will execute the question2Combined function, which runs both experiments and plots the combined results.

## Functions

- `softsvm(l, trainX, trainy)`: Function to train the Soft-SVM model.
- `custom_A_matrix(trainX, trainy)`: Function to construct the matrix A for the quadratic programming problem.
- `question2()`: Main function to choose between different experiments.
- `question2Combined()`: Function to run both experiments and plot the combined results.
- `question2firstExperement()`: Function to run the first experiment and plot its results.
- `question2secondExperement()`: Function to run the second experiment and plot its results.
- `experiment1(data)`: Function to perform the first experiment.
- `experiment2(data)`: Function to perform the second experiment.
- `plot_experiment1(...)`: Function to plot the results of the first experiment.
- `plot_experiment2(...)`: Function to plot the results of the second experiment.
- `simple_test()`: Function to perform a simple test of the softsvm function.

### Enjoy
