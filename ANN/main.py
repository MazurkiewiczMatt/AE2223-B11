import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

def get_file(file):
    # Returns the full path, if the file is in the same folder as the main .py program.
    # Useful if a computer uses some random directory (like mine)
    path = os.path.join(os.path.dirname(__file__), file)
    return path


def get_folder_file(folder, file):
    # Returns the full path, if the file is not in the same folder as the main .py program.
    # Useful if a computer uses some random directory (like mine)
    extension = os.path.join(folder, file)
    path = get_file(extension)
    return path

# Load the training and validation data
def load_data(file_):
    directory_ = get_file(file_)
    data = pd.read_csv(directory_)  #r"C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\x.csv"
    # print(data.head(n=1000))
    # at this point, there is 100 rows (1 row = 1 image) and 10002 columns (10000 pixels + label column + useless column)
    data = np.array(data)
    data = data.T
    # the first column is useless, so it is deleted
    data = np.delete(data, 0, 0)
    data = data.T
    m, n = data.shape
    data_pic = data
    # shuffling before splitting the data into validation and training subsets
    np.random.shuffle(data)

    data_valid = data[0:100-HIDDEN_LAYER].T
    Y_valid = data_valid[0]
    X_valid = data_valid[1:n]
    X_valid = X_valid / 255.

    data_train = data[100-HIDDEN_LAYER:m].T
    Y_train = data_train[0]
    print('Y_train ', Y_train)
    X_train = data_train[1:n]
    X_train = X_train / 255.
    dummy, m_train = X_train.shape
    return n, m, X_train, Y_train, X_valid, Y_valid

# Initializes the weights and bias for the input values
def init_values():
    W1 = np.random.rand(HIDDEN_LAYER, INPUT_LAYER) - 0.5
    b1 = np.random.rand(HIDDEN_LAYER, 1) - 0.5
    W2 = np.random.rand(OUTPUT_LAYER, HIDDEN_LAYER) - 0.5
    b2 = np.random.rand(OUTPUT_LAYER, 1) - 0.5
    return W1, b1, W2, b2

# Changes Z to probabilities
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# 
def cubic(Z):
    return np.power(Z, 3)

# Make a forward prediction by adding weighted sums of the parameters
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = max_val_function(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
# Returns 0 for everything below 0, and the value for everything above 0
# Is an activation function
def max_val_function(Z):
    return np.maximum(Z, 0)
    
# Derivative of the max_val_function:
def larger_val_function(Z):
    return Z > 0

# Output is the zero vector except a 1 for the correct parameter
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Check how good the predictions are and adjust the weights accordingly
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * larger_val_function(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update the values where necessary
def update_values(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Eliminates negative probabilities
def get_predictions(A2):
    return np.argmax(A2, 0)

# Number of correctly guesses parameters, normalised to probabilities
def accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Actual learning process / iteration of the ANN:
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_values()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_values(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 20 == 0:  # prints out every 20th iteration to see progress in the learning curve
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(accuracy(predictions, Y))
    return W1, b1, W2, b2

# Make a forward prediction by using forward propagation
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Function which predicts what a given image is
def test_prediction(index, W1, b1, W2, b2, X, Y):
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    
    names = ['A380', 'AN225', 'B747', 'B787', 'Beluga', 'C130', 'F16', 'Fokker', 'PHLAB', 'SS100']

    for p in range(10):
        if prediction == [p]:
            prediction = names[p]
        if label == p:
            label = names[p]
    '''if prediction == [0]:
        prediction = 'A380'
    elif prediction == [1]:
        prediction = 'AN225'
    elif prediction == [2]:
        prediction = 'B747'
    elif prediction == [3]:
        prediction = 'B787'
    elif prediction == [4]:
        prediction = 'Beluga'
    elif prediction == [5]:
        prediction = 'C130'
    elif prediction == [6]:
        prediction = 'F16'
    elif prediction == [7]:
        prediction = 'Fokker'
    elif prediction == [8]:
        prediction = 'PHLAB'
    elif prediction == [9]:
        prediction = 'SS100'
    '''

    '''if label == 0:
        label = 'A380'
    elif label == 1:
        label = 'AN225'
    elif label == 2:
        label = 'B747'
    elif label == 3:
        label = 'B787'
    elif label == 4:
        label = 'Beluga'
    elif label == 5:
        label = 'C130'
    elif label == 6:
        label = 'F16'
    elif label == 7:
        label = 'Fokker'
    elif label == 8:
        label = 'PHLAB'
    elif label == 9:
        label = 'SS100'
    '''

    print("Prediction: ", prediction)
    print("Real: ", label)


# layers:
INPUT_LAYER = 100*100  # 100x100 pixels
HIDDEN_LAYER = 90   # Hidden layer can maximally be ~99
OUTPUT_LAYER = 10

# Load the coefficients
n, m, X_train, Y_train, X_valid, Y_valid = load_data('x.csv')
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.01, 300)

# To test the ANN with test data
for i in range(100-HIDDEN_LAYER):
    test_prediction(i, W1, b1, W2, b2, X_valid, Y_valid)
