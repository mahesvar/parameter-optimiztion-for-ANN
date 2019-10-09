import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import os
os.chdir("C:/Users/mahesvar/Desktop/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Model_Selection")

dataset = pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values     # x is considered as matrix
y=dataset.iloc[:,-1].values

# split the data set into the training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25,random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
fc_x = StandardScaler()
x_train = fc_x.fit_transform(x_train)
x_test = fc_x.transform(x_test)


def custom_model(optimizer, no_of_layers = 1, nodes = 9):
    # Initializing ANN 
    model = Sequential()
    # Adding Input Layer and the First Hidden Layer 
    # Thumb rule: First output_dim = (number of IV + Number of Classifiers) / 2
    model.add(Dense(units = nodes, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
 
    # Adding hidden layers depends on no_of_layers argument
    for i in range(no_of_layers):
        model.add(Dense(units = nodes, kernel_initializer = 'uniform', activation = 'relu'))
 
    # Adding the final output layer (units would be max of category DV value + 1 
    # Categories are [6, 5, 7, 8, 4, 3, 9] so units = 10
    # activation should be softmax for Multi-Classifier
    model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))
 
    # Compiling ANN 
    # For Multi-Classifier loss should be sparse_categorical_crossentropy
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    return model
 
 
from keras.callbacks import TensorBoard
#import threading
 
# TensorBoard Callback
cb = TensorBoard()
# Different lists of tuning params
optmz_list   = ["adam", "sgd"]
layers_list  = [1, 2, 3, 4, 5]
nodes_list   = [8, 9, 10]
# Batch size and epochs
batch_size   = 5
epochs       = 100
# Threshold accuracy 70% 
min_accuracy = 0.70
 
# Holding all the ANN model threads 
model_thread_list = []
 
 
# Looping through different values of optimizer, number of layers and nodes
for opt in optmz_list:
    for layer in layers_list:
        for nodes in nodes_list:
            model = custom_model(optimizer = opt, no_of_layers = layer, nodes = nodes)
            # Fitting ANN to the Training Set
            model.fit(x_train, 
                      y_train,
                      batch_size = 5,
                      epochs = epochs,
                      verbose = 0,
                      validation_data = (x_test, y_test),
                      callbacks = [cb])
            # Calculate the accuracy score
            final_score = model.evaluate(x_test, y_test, verbose = 0)
            formatted_accuracy = '{:2.2%}'.format(final_score[1])
            # Only print the model details if the accuracy is more than min_accuracy
            if final_score[1] > min_accuracy:
                print("ANN Model--Optimizer = {}, Hidden Layers = {}, Neurons = {}".format(opt, layer, nodes))
                print("-------------------------------------------------------------------------------------")
                print('Test Loss:', final_score[0])
                print('Test Accuracy:', formatted_accuracy)
                print("=====================================================================================")
