import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import cvxopt
from numpy import double


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    m, d = trainX.shape

    # Define the matrices and vectors for the quadratic programming problem
    H = np.zeros((m+d ,m+d))
    H[:d, :d] = 2 * l * np.eye(d)

    u_d = np.zeros((d, 1))
    u_m = np.ones((m, 1)) / m
    u = np.concatenate((u_d, u_m), axis=0)

    v_m2 = np.zeros((m, 1))
    v_m1 = np.ones((m, 1))
    v = np.concatenate((v_m1, v_m2), axis=0)

    A = custom_A_matrix(trainX , trainy)

    # Solve the quadratic programming problem
    sol = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(u), -A, cvxopt.matrix(-v))

    # Extract the solution vector
    w = np.array(sol['x'][:d])

    return w

def custom_A_matrix(trainX, trainy):
    m, d = trainX.shape

    # Constructing the first block Y * X
    YmatXmat = np.diag(trainy) @ trainX


    # Augment the first block with the identity matrix I_m
    YmatXmat_augmented = np.hstack((YmatXmat, np.eye(m)))


    # Constructing the second block [0_{m x d}, I_m]
    second_block = np.hstack((np.zeros((m, d)), np.eye(m)))

    # Concatenate the two blocks vertically
    A = np.vstack((YmatXmat_augmented, second_block))

    return cvxopt.matrix(A)



def question2():
    # choose between which question do you want 
    #question2firstExperement()
    #question2secondExperement()
    question2Combined()
    
    
    
    
    
def question2Combined():
    data = np.load('EX2q2_mnist.npz')

    # Experiment 1
    x_exp1, y_train_exp1, down_train_exp1, up_train_exp1, y_test_exp1, down_test_exp1, up_test_exp1 = experiment1(data)

    # Experiment 2
    x_exp2, sample_error_train_exp2, sample_error_test_exp2 = experiment2(data)

    # Plotting combined results
    plt.figure(figsize=(10, 6))
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Error")
    plt.xlim((0, 10.5))
    plt.ylim((0, 0.7))
    plt.title("Combined Soft-SVM Experiment Results")

    # Plotting Experiment 1
    plt.errorbar(x_exp1, y_train_exp1, yerr=[down_train_exp1, up_train_exp1], capsize=3, fmt="--o", color="yellow",
                 ecolor="yellow", label="Train Error Experiment 1")
    plt.errorbar(x_exp1, y_test_exp1, yerr=[down_test_exp1, up_test_exp1], capsize=3, fmt="--o", color='green',
                 ecolor="green", label="Test Error Experiment 1")

    # Plotting Experiment 2
    plt.scatter(x_exp2, sample_error_train_exp2, label="Train Error Experiment 2", color="orange", edgecolors='black')
    plt.scatter(x_exp2, sample_error_test_exp2, label="Test Error Experiment 2", color="blue", edgecolors='black')

    plt.legend()
    plt.show()

    
    
    
    
def question2firstExperement():
    data = np.load('EX2q2_mnist.npz')
    # Experiment 1
    x_exp1, y_train_exp1, down_train_exp1, up_train_exp1, y_test_exp1, down_test_exp1, up_test_exp1 = experiment1(data)
    plot_experiment1(x_exp1, y_train_exp1, down_train_exp1, up_train_exp1, y_test_exp1, down_test_exp1, up_test_exp1)

def question2secondExperement():
    data = np.load('EX2q2_mnist.npz')
    # Experiment 2
    x_exp2, sample_error_train_exp2, sample_error_test_exp2 = experiment2(data)
    plot_experiment2(x_exp2, sample_error_train_exp2, sample_error_test_exp2)
    
    



def experiment1(data):
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']

    # Experiment 1
    # Parameters for Experiment 1
    n_values_exp1 = list(range(1, 11))
    num_of_tests = 10
    m_exp1 = 100

    # Soft-SVM algorithm for Experiment 1

    # Matrices to save the experiment results
    sample_error_train_exp1 = np.zeros((len(n_values_exp1), num_of_tests))
    sample_error_test_exp1 = np.zeros((len(n_values_exp1), num_of_tests))

    # Iterate over different values of lambda
    for n in n_values_exp1:
        lamda = 10 ** n
        for k in range(num_of_tests):
            # Randomly select m training examples
            train_indices = np.random.permutation(trainX.shape[0])[:m_exp1]
            train_X_subset = trainX[train_indices]
            train_Y_subset = trainY[train_indices]

            # Run softsvm algorithm
            w = softsvm(lamda, train_X_subset, train_Y_subset)

            # Calculate errors for training and test sets
            train_predictions = np.sign(train_X_subset @ w)
            sample_error_train_exp1[n-1][k] = np.mean(train_Y_subset != np.concatenate(train_predictions))

            test_predictions = np.sign(testX @ w)
            sample_error_test_exp1[n-1][k] = np.mean(testY != np.concatenate(test_predictions))

    # Calculate statistics for Experiment 1
    sample_statistics_train_exp1 = {
        "avg": [np.mean(error) for error in sample_error_train_exp1],
        "max": [np.max(error) for error in sample_error_train_exp1],
        "min": [np.min(error) for error in sample_error_train_exp1]
    }
    sample_statistics_test_exp1 = {
        "avg": [np.mean(error) for error in sample_error_test_exp1],
        "max": [np.max(error) for error in sample_error_test_exp1],
        "min": [np.min(error) for error in sample_error_test_exp1]
    }
    x_exp1 = n_values_exp1
    y_train_exp1 = sample_statistics_train_exp1["avg"]
    y_test_exp1 = sample_statistics_test_exp1["avg"]
    up_train_exp1 = [item1 - item2 for item1, item2 in zip(sample_statistics_train_exp1["max"], y_train_exp1)]
    down_train_exp1 = [item1 - item2 for item1, item2 in zip(y_train_exp1, sample_statistics_train_exp1["min"])]
    up_test_exp1 = [item1 - item2 for item1, item2 in zip(sample_statistics_test_exp1["max"], y_test_exp1)]
    down_test_exp1 = [item1 - item2 for item1, item2 in zip(y_test_exp1, sample_statistics_test_exp1["min"])]
    
    return x_exp1, y_train_exp1, down_train_exp1, up_train_exp1, y_test_exp1, down_test_exp1, up_test_exp1


def experiment2(data) : 
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']
    
    # Experiment 2
    m_exp2 = 1000
    n_values_exp2 = [1, 3, 5, 8]

    # Soft-SVM algorithm for Experiment 2
    sample_error_train_exp2 = np.zeros(len(n_values_exp2))
    sample_error_test_exp2 = np.zeros(len(n_values_exp2))

    for i in range(len(n_values_exp2)):
        lamda = 10 ** n_values_exp2[i]
        train_indices_exp2 = np.random.permutation(trainX.shape[0])[:m_exp2]
        train_X_subset_exp2 = trainX[train_indices_exp2]
        train_Y_subset_exp2 = trainY[train_indices_exp2]

        w = softsvm(lamda, train_X_subset_exp2, train_Y_subset_exp2)

        train_predictions_exp2 = np.sign(train_X_subset_exp2 @ w)
        sample_error_train_exp2[i] = np.mean(train_Y_subset_exp2 != np.concatenate(train_predictions_exp2))

        test_predictions_exp2 = np.sign(testX @ w)
        sample_error_test_exp2[i] = np.mean(testY != np.concatenate(test_predictions_exp2))
        
    x_exp2 = n_values_exp2
    return x_exp2, sample_error_train_exp2, sample_error_test_exp2
    

    
    
def plot_experiment1(x_exp1, y_train_exp1, down_train_exp1, up_train_exp1, y_test_exp1, down_test_exp1, up_test_exp1):
    # Plotting Experiment 1
    plt.figure(figsize=(10, 6))
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Error")
    plt.xlim((0, 10.5))
    plt.ylim((0, 0.7))
    plt.title("Soft-SVM Experiment 1 Results")
    
    plt.errorbar(x_exp1, y_train_exp1, yerr=[down_train_exp1, up_train_exp1], capsize=3, fmt="--o", color="yellow",
                 ecolor="yellow", label="Train Error Experiment 1")
    plt.errorbar(x_exp1, y_test_exp1, yerr=[down_test_exp1, up_test_exp1], capsize=3, fmt="--o", color='green',
                 ecolor="green", label="Test Error Experiment 1")

    plt.legend()
    plt.show()

def plot_experiment2(x_exp2, sample_error_train_exp2, sample_error_test_exp2):
    # Plotting Experiment 2
    plt.figure(figsize=(10, 6))
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Error")
    plt.title("Soft-SVM Experiment 2 Results")

    plt.scatter(x_exp2, sample_error_train_exp2, label="Train Error Experiment 2", color="yellow", edgecolors='black')
    plt.scatter(x_exp2, sample_error_test_exp2, label="Test Error Experiment 2", color="green", edgecolors='black')

    plt.legend()
    plt.show()



def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    question2()

    # here you may add any code that uses the above functions to solve question 2
