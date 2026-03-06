import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * ( 1 - x)

training_input = np.array([[0,2,1,0],
                           [0,4,6,3],
                           [1,5,9,1],
                           [2,6,7,1]])

training_output = np.array([[3,5,7,2]]).T / 10

np.random.seed(1)

synaptic_weights = np.random.random((4,1)) - 1

print("training inputs shape: ", training_input, training_input.shape)
print("training outputs: ", training_output, training_output.shape)
print("Random synaptic_weights shape: ", synaptic_weights, synaptic_weights.shape)

learning_rate = 0.1

for iterations in range(100000):
    input_layer = training_input
    
    output = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_output - output
    
    adjustment = error * sigmoid_derivative(output)
    
    synaptic_weights += learning_rate * np.dot(input_layer.T, adjustment)
    
print("synaptic_weights: ", synaptic_weights)
print("Output: ", output)
    
new_input_layer = np.array([[0,2,1,0],
                            [0,4,6,3],
                            [1,5,9,1],
                            [2,6,7,1]])

new_synaptic_weights = np.array([[-0.43691485],
                             [-0.69825462],
                             [ 0.54921138],
                             [-0.1674166 ]]) 

output = sigmoid(np.dot(new_input_layer, new_synaptic_weights))
print(output * 10)
