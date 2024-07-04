import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Sets common hyperparameters
learning_rate = 0.001
batch_size = 128
epochs = 10

def create_and_train_network(num_hidden_layers, num_neurons):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(784,), kernel_initializer=he_normal()))
    
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(num_neurons, activation='relu', kernel_initializer=he_normal()))
    
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Evaluate the model on the test data
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    return model, accuracy


def create_and_train_network(num_hidden_layers, num_neurons):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(784,), kernel_initializer=he_normal()))
    
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(num_neurons, activation='relu', kernel_initializer=he_normal()))
    
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Evaluate the model on the test data
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    return model, accuracy


print("Case 1:")
# Train networks with different depths and calculate accuracie
depths = [2, 4, 8]
num_neurons = 50

for depth in depths:
    model, accuracy = create_and_train_network(depth, num_neurons)
    print(f"Network with {depth} hidden layers and {num_neurons} neurons: Accuracy = {accuracy:.4f}")


def perturb_and_evaluate(model, layer_index):
    perturbed_model = tf.keras.models.clone_model(model)
    perturbed_model.set_weights(model.get_weights())  # Copy weights from the original model

    # Perturb the weights of the specified layer
    original_weights = perturbed_model.layers[layer_index].get_weights()
    perturbed_weights = [w + np.random.normal(0, np.sqrt(2 / original_weights[0].shape[0]), w.shape) for w in original_weights]
    perturbed_model.layers[layer_index].set_weights(perturbed_weights)

    perturbed_model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


    # Evaluate the perturbed model and calculate deviation in accuracy
    perturbed_accuracy = perturbed_model.evaluate(x_test, y_test, verbose=0)[1]
    deviation = model.evaluate(x_test, y_test, verbose=0)[1] - perturbed_accuracy
    return deviation, layer_index


print("\nCase 2:")
for depth in depths:
    model, _ = create_and_train_network(depth, num_neurons)
    print(f"Network with {depth} hidden layers and {num_neurons} neurons:")
    deviations = []

    # Perturb each hidden layer and evaluate performance deviation
    for i in range(1, depth):
        deviation, layer_index = perturb_and_evaluate(model, i)
        deviations.append((deviation, layer_index))

    # Rank layers based on performance deviation (in descending order)
    deviations.sort(reverse=True)
    for deviation, layer_index in deviations:
        print(f"  Layer {layer_index} - Performance deviation: {deviation:.4f}")
