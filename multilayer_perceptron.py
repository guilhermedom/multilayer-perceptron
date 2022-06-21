# -*- coding: utf-8 -*-

import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class MLP(object):
    def __init__(self, iterations = 15, learning_rate = 0.001, momentum = 0.9, nn_input = 11, nn_hidden = 2, nn_output = 3):
        self.nn_input = nn_input
        self.nn_hidden = nn_hidden
        self.nn_output = nn_output

        self.learning_rate = learning_rate 
        self.reg_param = 0
        self.iterations = iterations
        self.m = 1599
        self.momentum = momentum

        self.W1 = np.random.normal(0, 0.1, (nn_hidden, nn_input))
        self.W2 = np.random.normal(0, 0.1, (nn_output, nn_hidden))
        
        self.B1 = np.random.normal(0, 0.1, (nn_hidden, 1))
        self.B2 = np.random.normal(0, 0.1, (nn_output, 1))

    def sigmoid(self, z, derivative=False):
        if derivative:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))

    def activation(self, x):
        input = x.reshape(x.shape[0], 1)

        sum1 = self.W1.dot(input) + self.B1
        activation1 = self.sigmoid(sum1)

        sum2 = self.W2.dot(activation1) + self.B2
        activation2 = self.sigmoid(sum2)

        return activation2
        
    
    def predict(self, X):
        self.m = X.shape[0]
        act = self.activation(X)
        #return np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(act == np.amax(act))[0]
        
    def fit(self, X, y):
        self.m = X.shape[0]
        
        delta_W1 = 0
        delta_W2 = 0
        delta_B1 = 0
        delta_B2 = 0

        self.cost = np.zeros((self.iterations, 1))
        
        for i in range(self.iterations):
            delta_W1 = 0
            delta_W2 = 0
            delta_B1 = 0
            delta_B2 = 0
            
            total_cost = 0
            
            for j in range(self.m):
            
                # propagation
                input = X[j].reshape(X[j].shape[0], 1)
                #print (X[j].reshape(X[j].shape[0], 1))

                sum1 = self.W1.dot(input) + self.B1
                activation1 = self.sigmoid(sum1)

                sum2 = self.W2.dot(activation1) + self.B2
                activation2 = self.sigmoid(sum2)

                # back propagation
                #if (y[j] == 0)
                delta_sum2 = activation2 - y[j]
                
                delta_W2 += delta_sum2 * activation1.T

                delta_sum1 = np.multiply((self.W2.T.dot(delta_sum2)), self.sigmoid(activation1, derivative=True))
                delta_W1 += delta_sum1.dot(input.T)

                delta_B1 += delta_sum1
                delta_B2 += delta_sum2

                # accumulates cost
                total_cost = total_cost + (-(y[j] * np.log(activation2)) - ((1 - y[j]) * np.log(1 - activation2)))
                
            # weights updates
            self.W1 = self.W1 - self.momentum * self.learning_rate * (delta_W1 / self.m)
            self.W2 = self.W2 - self.momentum * self.learning_rate * (delta_W2 / self.m)

            self.B1 = self.B1 - self.learning_rate * (delta_B1 / self.m)
            self.B2 = self.B2 - self.learning_rate * (delta_B2 / self.m)
            self.cost[i] = abs((total_cost[0] / self.m))
  
class MLP_regressor(object):
    def __init__(self, iterations = 15, learning_rate = 0.001, momentum = 0.9, nn_input = 11, nn_hidden = 2, nn_output = 1):
        self.nn_input = nn_input
        self.nn_hidden = nn_hidden
        self.nn_output = nn_output

        self.learning_rate = learning_rate 
        self.reg_param = 0
        self.iterations = iterations
        self.m = 1599
        self.momentum = momentum

        self.W1 = np.random.normal(0, 0.1, (nn_hidden, nn_input))
        self.W2 = np.random.normal(0, 0.1, (nn_output, nn_hidden))
    
        self.B1 = np.random.normal(0, 0.1, (nn_hidden, 1))
        self.B2 = np.random.normal(0, 0.1, (nn_output, 1))

    def sigmoid(self, z, derivative=False):
        if derivative: 
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))

    def activation(self, x):
        input = x.reshape(x.shape[0], 1)

        sum1 = self.W1.dot(input) + self.B1
        activation1 = self.sigmoid(sum1)

        sum2 = self.W2.dot(activation1) + self.B2
        activation2 = self.sigmoid(sum2)

        return activation2
    
  
    def predict(self, X):
        self.m = X.shape[0]
        return (self.activation(X))
    
    
    def fit(self, X, y):
        self.m = X.shape[0]
    
        delta_W1 = 0
        delta_W2 = 0
        delta_B1 = 0
        delta_B2 = 0

        self.cost = np.zeros((self.iterations, 1))
    
        for i in range(self.iterations):
            delta_W1 = 0
            delta_W2 = 0
            delta_B1 = 0
            delta_B2 = 0
        
            total_cost = 0
        
            for j in range(self.m):
          
                # propagation
                input = X[j].reshape(X[j].shape[0], 1)
                #print (X[j].reshape(X[j].shape[0], 1))

                sum1 = self.W1.dot(input) + self.B1
                activation1 = self.sigmoid(sum1)

                sum2 = self.W2.dot(activation1) + self.B2
                activation2 = self.sigmoid(sum2)

                # back propagation
                #if (y[j] == 0)
                delta_sum2 = activation2 - y[j]
                
                delta_W2 += delta_sum2 * activation1.T

                delta_sum1 = np.multiply((self.W2.T.dot(delta_sum2)), self.sigmoid(activation1, derivative=True))
                delta_W1 += delta_sum1.dot(input.T)

                delta_B1 += delta_sum1
                delta_B2 += delta_sum2

                # accumulates cost
                total_cost = total_cost + (-(y[j] * np.log(activation2)) - ((1 - y[j]) * np.log(1 - activation2)))
            
            # weights updates
            self.W1 = self.W1 - self.momentum * self.learning_rate * (delta_W1 / self.m)
            self.W2 = self.W2 - self.momentum * self.learning_rate * (delta_W2 / self.m)

            self.B1 = self.B1 - self.learning_rate * (delta_B1 / self.m)
            self.B2 = self.B2 - self.learning_rate * (delta_B2 / self.m)
            self.cost[i] = (total_cost / self.m)
        
def accuracy_score(pred_y, test_y):
    acc_counter = 0

    for i in range(0, pred_y.shape[0]):
        if pred_y[i] == test_y[i]:
            acc_counter += 1
    return (1/(pred_y.shape[0]) * acc_counter)

def train_test_split(dataset, percentage):
    np.random.shuffle(dataset)
    
    index_train_x = math.floor(percentage * dataset.shape[0])
    #index_train_y = dataset.shape[0] - math.floor(percentage * dataset.shape[0])
    
    train = dataset[:index_train_x, :]
    train_y = train[:, -1]
    train_x = train[:, :-1]
    
    test = dataset[index_train_x:, :]
    test_y = test[:, -1]
    test_x = test[:, :-1]

    return (train_x, train_y, test_x, test_y)

# Load and preprocess data for classification
df = pd.read_csv('winequality-red.csv')

labels = df.iloc[:, -1]
label_types = np.unique(labels)

feature_list = list(df.columns)

# Clean data by removing unknown, NaN and duplicate values
df = df.replace('?', np.nan)
df = df.replace('Bad', 1)
df = df.replace('Mid', 2)
df = df.replace('Good', 3)
df = df.dropna()
df = df.drop_duplicates()
dataset = df.to_numpy()

dataset_without_ind = dataset[:, 1:]
X = dataset_without_ind[:, :-1]
y = dataset[:, -1]

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Load and preprocess data for regression
df = pd.read_csv('default_features_1059_tracks.txt')

# Clean data by removing unknown, NaN and duplicate values
df = df.replace('?', np.nan)
df = df.dropna()
df = df.drop_duplicates()
dataset = df.to_numpy()

#print(feature_list)

X = dataset_without_ind[:, :-1]
y = dataset[:, -1]

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# divisão da base de dados em conjunto de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25)

model = MLP(iterations = 100, learning_rate = 0.001, momentum=0.9, nn_hidden=10)
model.fit(x_train, y_train)

#acc = model.predict(x_test)
acc = []
for sample in x_test:
    #print("\n")
    #print("Sample: ", sample)
    #print("Predicted value: ", model.predict(sample))
    acc.append(model.predict(sample))

print(accuracy_score(np.array(acc), y_test))

plt.plot(range(model.iterations), model.cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size= 1 - p)

model = MLP_regressor(iterations = 1000, learning_rate = 0.001, momentum=0.1)
model.fit(x_train, y_train)

#acc = model.predict(x_test)
acc = []
for sample in x_test:
    #print("\n")
    #print("Sample: ", sample)
    #print("Predicted value: ", model.predict(sample))
    acc.append(model.predict(sample))

plt.plot(range(model.iterations), model.cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()