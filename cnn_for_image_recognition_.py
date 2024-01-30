"""CNN for image recognition..ipynb

 
Build a CNN for image recognition.


 1. Data preparation

  1.1. Load data
"""

# Load Cifar-10 Data
import keras
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('shape of x_train: ' + str(x_train.shape))
print('shape of y_train: ' + str(y_train.shape))
print('shape of x_test: ' + str(x_test.shape))
print('shape of y_test: ' + str(y_test.shape))
print('number of classes: ' + str(np.max(y_train) - np.min(y_train) + 1))

''' 1.2. One-hot encode the labels'''



def to_one_hot(y, num_class=10):
    y = np.array(y).reshape(-1)
    n = y.shape[0]
    y_one_hot = np.zeros((n, num_class))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot

y_train_vec = to_one_hot(y_train)
y_test_vec = to_one_hot(y_test)

print('Shape of y_train_vec: ' + str(y_train_vec.shape))
print('Shape of y_test_vec: ' + str(y_test_vec.shape))

print(y_train[0])
print(y_train_vec[0])

""" 1.3. Randomly partition the training set to training and validation sets

Randomly partition the 50K training samples to 2 sets:
* a training set containing 40K samples: x_tr, y_tr
* a validation set containing 10K samples: x_val, y_val

"""

# Shuffle the indices of the training set
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

# Split the indices into training indices and validation indices
train_indices, val_indices = indices[:40000], indices[40000:]

# Use the indices to extract the corresponding samples and labels
x_tr, y_tr = x_train[train_indices], y_train_vec[train_indices]
x_val, y_val = x_train[val_indices], y_train_vec[val_indices]

print('Shape of x_tr: ' + str(x_tr.shape))
print('Shape of y_tr: ' + str(y_tr.shape))
print('Shape of x_val: ' + str(x_val.shape))
print('Shape of y_val: ' + str(y_val.shape))

'''2. Build a CNN and tune its hyper-parameters'''

# Build the model
from keras import models,layers
# Define the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # first hidden layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (4, 4), activation='relu')) # second layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten()) # bringing it to 1D vector
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()   #printing Neural Network Structure

from keras import optimizers
# Define the optimization algorithm
opt = optimizers.Adagrad(learning_rate = 0.01)
# Compile the model
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

"""Here I used diffrent optimizers such as Adamax, SGD, Adam, etc. But Adagard gave the best results. I also changed and tried diffrent learning rates and 0.01 gave the best accuracy and low loss."""

# Train the model and store model parameters/loss values
history = model.fit(x_tr, y_tr, batch_size=128, epochs=100, validation_data=(x_val, y_val))

print(history.history)

""" 3. Plot the training and validation loss curve versus epochs."""

# Plot the loss curve
from matplotlib import pyplot as plt
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(100),train_loss, 'red',label = 'Training Loss')
plt.plot(range(100),val_loss, 'blue',label = 'Validation Loss')
plt.show()

"""4. Train (again) and evaluate the model

Train the model on the entire training set

Why? Previously, you used 40K samples for training; you wasted 10K samples for the sake of hyper-parameter tuning. Now you already know the hyper-parameters, so why not using all the 50K samples for training?
"""

#<Compile your model again (using the same hyper-parameters you tuned above)>
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

#<Train your model on the entire training set (50K samples)>
h_train = model.fit(x_train, y_train_vec, batch_size=128, epochs=100)

""" 5. Evaluate the model on the test set

Do NOT use the test set until now. Make sure that your model parameters and hyper-parameters are independent of the test set.
"""

# Evaluate your model performance (testing accuracy) on testing data.
h_pred = model.evaluate(x_test,y_test_vec)

""" 6. Building model with new structure
-build another model with adding new layers (e.g, BN layer or dropout layer, ...).

-comparing their loss curve and testing accuracy and analyze your findings.

"""

new_model = models.Sequential()
new_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # first hidden layer
new_model.add(layers.MaxPooling2D((2, 2)))
new_model.add(layers.Dropout(0.2)) #adding dropout layer
new_model.add(layers.Conv2D(64, (4, 4), activation='relu')) # second layer
new_model.add(layers.MaxPooling2D((2, 2)))
new_model.add(layers.Flatten()) # bringing it to 1D vector
new_model.add(layers.Dropout(0.2)) #adding dropout layer
new_model.add(layers.Dense(256, activation='relu'))
new_model.add(layers.Dropout(0.2)) #adding dropout layer
new_model.add(layers.Dense(10, activation='softmax'))

new_model.summary()

new_model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

h_new_train =new_model.fit(x_train, y_train_vec, batch_size=128, epochs=100)

new_model.evaluate(x_test,y_test_vec)

new_model_2 = models.Sequential()
new_model_2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # first hidden layer
new_model_2.add(layers.BatchNormalization()) #BN layer
new_model_2.add(layers.MaxPooling2D((2, 2)))
new_model_2.add(layers.BatchNormalization())
new_model_2.add(layers.Conv2D(64, (4, 4), activation='relu')) # second layer
new_model_2.add(layers.BatchNormalization())
new_model_2.add(layers.MaxPooling2D((2, 2)))
new_model_2.add(layers.BatchNormalization()) #BN Layer
new_model_2.add(layers.Flatten()) # bringing it to 1D vector
new_model_2.add(layers.Dense(256, activation='relu'))
new_model_2.add(layers.Dense(10, activation='softmax'))

new_model_2.compile(opt, loss='categorical_crossentropy', metrics=['accuracy']) #compiling the model

h_new_2_train =new_model_2.fit(x_train, y_train_vec, batch_size=128, epochs=100)  #training

new_model_2.evaluate(x_test,y_test_vec)
