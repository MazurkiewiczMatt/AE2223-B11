import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# HYPERPARAMETERS

# Test Data
'''
INPUT_LAYER = 28*28
HIDDEN_LAYER = 10
OUTPUT_LAYER = 10
'''

INPUT_LAYER = 100*100
HIDDEN_LAYER = 40   # Hidden layer can maximally be ~99
OUTPUT_LAYER = 10

data = pd.read_csv(r"C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\x.csv")
#print(data.head(n=1000)) 
#at this point, there is 100 rows (1 row = 1 image) and 10002 columns (10000 pixels + label column + useless column)
data = np.array(data)
data = data.T
data = np.delete(data, 0, 0)   # the first column is useless, so delete it
data = data.T
m, n = data.shape
data_pic = data
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# from here
data_dev = data[0:100-HIDDEN_LAYER].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[100-HIDDEN_LAYER:m].T
Y_train = data_train[0]
print('Y_train ', Y_train)
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
# till there makes my brain hurt
# TO DO: adapt that so it properly gets a label and the data

def init_params():
    W1 = np.random.rand(HIDDEN_LAYER, INPUT_LAYER) - 0.5
    b1 = np.random.rand(HIDDEN_LAYER, 1) - 0.5
    W2 = np.random.rand(OUTPUT_LAYER, HIDDEN_LAYER) - 0.5
    b2 = np.random.rand(OUTPUT_LAYER, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return Z**2
    #return np.maximum(Z, 0)

'''
def Quadratic(Z):
    return Z**2 + Z + np.eye(m)

def Quadratic_deriv(Z):
    return 2 * Z + np.eye(m)

def Cubic(Z):
    return Z**3 + Z**2 + Z + np.eye(m)

def Cubic_deriv(Z):
    return 3 * Z**2 + Z + np.eye(m)
'''

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1    # Z1 is almost always negative, so ReLU returns 0, therefore it won't learn
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return 2 * Z
    #return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))   # Y.max() + 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y    # fine
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

'''running = True
while running:
    try:
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)
        running = False
    except:
        running = True
        print("It broke")'''
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.01, 300)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    if prediction == [0] :
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
    
    if label == 0 :
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

    print("Prediction: ", prediction)
    print("Label: ", label)
    
    '''
    current_image = current_image.reshape((100, 100)) * 255
    plt.gray()
    plt.imshow(current_image)
    plt.show()
    '''

test_prediction(9 ,W1, b1, W2, b2)