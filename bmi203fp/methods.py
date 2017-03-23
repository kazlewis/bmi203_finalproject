import numpy as np
import random

def sigmoid(x): # Definition of sigmoid
    return 1 / (1 + np.exp(-x))

def dsigmoid(y): # definition of derivative of sigmoid, in a format to solve incompatability issues with numpy matricies
    return np.multiply(y, 1.0 - y)

def train_autoencoder(input_data, desired_output, input_layer_size, hidden_layer_size, output_layer_size, L, alpha, iterations):
    '''
    Initial method to train and validate the neural network as an auto encoder by updating weight and bias matrices between layers
    
    Inputs: training / input data (as matrix); desired output (as matrix); size of input, hidden, output layers
            weight decay parameter lambda (L); learning rate alpha (a); total iterations 
    Outputs: final output layer, average sum of errors
    '''
    np.random.seed(4) # Keep random seed constant for testing purposes
    m = input_data.shape[0] # number of training examples
    
    # Initialize random weights with mean 0
    input_weights = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
    hidden_weights = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1
    
    # Initialize bias 'matrices' of zeros
    input_bias = np.zeros((input_layer_size,hidden_layer_size))
    hidden_bias = np.zeros((input_layer_size,output_layer_size)) 
    
    # Iterate to train weight matrices
    for i in range(0, iterations):
        
        # input data = input layer; contains all training data, simultaneously
        input_layer = input_data
        
        # Given the training data and weight matrix, tranform via the dot product, add bias matrix
        input_transform = np.dot(input_layer, input_weights) + input_bias
        # Pass through the activation function
        hidden_layer = sigmoid(input_transform)
        
        # Again, add bias and propagate predicted values through the weights between hidden + output layers
        hidden_transform = np.dot(hidden_layer, hidden_weights) + hidden_bias
        output_layer = sigmoid(hidden_transform)
    
        # Get the difference between the desired output and the result from the autoencoder
        output_layer_error = desired_output - output_layer
        
        # Print the error every so often 
        if (i == 0 or (i % (iterations / 10)) == 0):
            print('Iteration: ' + str(i) + '\n\tAverage Error: ' + str(np.mean(np.abs(output_layer_error))) + '\n')
            
        # Take the error weighted derivative to avoid changing weights that lead to post-sigmoid values close to either 0 or 1 
        # Because these are probably starting to converge -> focus effort on changing ambiguous (~ 0.5) values
        output_layer_delta = output_layer_error * dsigmoid(output_layer)
    
        # Backpropagate to determine the error in the hidden layer
        hidden_layer_error = output_layer_delta.dot(hidden_weights.T)
        
        # Another error-weighted derivative
        hidden_layer_delta = hidden_layer_error * dsigmoid(hidden_layer)
    
        # Update the weights for each layer in batch
        hidden_weights += alpha * (((1 / m ) * (hidden_layer.T.dot(output_layer_delta))) + L * hidden_weights)
        hidden_bias += (alpha / m) * output_layer_delta
        
        input_weights += alpha * (((1 / m ) * (input_layer.T.dot(hidden_layer_delta))) + L * input_weights)
        input_bias += (alpha / m) * hidden_layer_delta
        
    print('After training the encoder for ' + str(iterations) + ' iterations, \n\tAverage Error: ' + str(np.mean(np.abs(output_layer_error))))
    print('input data: ')
    print(input_data)
    print('output data: ')
    print(output_layer)
    return output_layer, np.mean(np.abs(output_layer_error))


def train_neuralnet(pos_data, neg_data, input_layer_size, hidden_layer_size, L, alpha, iterations, convergence_criteria, print_output = False):
    '''
    Method to train the neural network using both positive and negative training data.
    For a given number of iterations, update weight and bias matrices between layers.
    Serves as a wrapper function for the actual unput of data into the network as defined by its weight and bias matrices.
    
    Inputs: positive / negative training data; size of input, hidden, output layers
            weight decay parameter lambda (L); learning rate alpha (a); total iterations; convergence criteria delta (improvement in average error)
    Outputs: avg positive error, avg negative error, input & hidden weight matricies, input and hidden bias matrices
    '''
    np.random.seed(4) # Keep random seed constant for testing purposes
    m = pos_data.shape[0] # number of training examples (assume for this case, equal amount of positive and negative data)
    
    # Initialize random weights with mean 0
    input_weights = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
    hidden_weights = 2 * np.random.random((hidden_layer_size, 1)) - 1 # Output layer size is always 1
    
    # Initialize bias nodes
    input_bias = np.zeros((1,hidden_layer_size)) # One input at a time
    hidden_bias = np.zeros((1, 1)) # Output layer size 1
    
    # Keep track of running error
    pos_error = 1.0
    neg_error = 1.0
    
    # Train network one data point at a time, and alternate between positive and negative examples
    for i in range(0, iterations):
        training_order = random.sample(range(0,m), m)  # pick random order to train each iteration
        
        # Keep track of error accrued every iteration
        pos_errors = [] 
        neg_errors = []
        
        for j in training_order: # Train with both positive and negative data iteratively for entirety of both datasets
            sample_input = np.asmatrix(pos_data[j]) # Test one positive input
            last_pos_error, input_weights, hidden_weights, input_bias, hidden_bias = NN_input(sample_input, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha, True, True)

            sample_input = np.asmatrix(neg_data[j]) # Test one negative input
            last_neg_error, input_weights, hidden_weights, input_bias, hidden_bias = NN_input(sample_input, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha, False, True)                
            
            pos_errors.append(last_pos_error)
            neg_errors.append(last_neg_error)
        
        # Calculate average errors, delta from training cycle
        avg_pos_error = sum(pos_errors) / float(len(pos_errors))
        avg_neg_error = sum(neg_errors) / float(len(neg_errors))
        pos_delta = np.abs(avg_pos_error - pos_error) / pos_error
        neg_delta = np.abs(avg_neg_error - neg_error) / neg_error
        
        if print_output and (i == 0 or (i % (iterations / 100)) == 0):  # Print progress every so often
            print('Iteration: ' + str(i) + '\n\tAvg Pos Error: ' + str(avg_pos_error) + '\n\tAvg Neg Error: ' + str(avg_neg_error))
            print('\tPos Err Delta: ' + str(pos_delta))
            print('\tNeg Err Delta: ' + str(neg_delta))
                
        if print_output and pos_delta <= convergence_criteria and neg_delta <= convergence_criteria: # Successive iterations have marginal gains; stop training
            print('Convergence criteria of ' + str(convergence_criteria) + ' met at iteration ' + str(i))
            print('Average Positive Error: ' + str (avg_pos_error))
            print('Average Negative Error: ' + str (avg_neg_error))
            return avg_pos_error, avg_neg_error, input_weights, hidden_weights, input_bias, hidden_bias
        else: # Keep going until convergence or iterations are complete
            pos_error = avg_pos_error
            neg_error = avg_neg_error
        
    return pos_error, neg_error, input_weights, hidden_weights, input_bias, hidden_bias
      
      
def test_neuralnet(pos_data, neg_data, input_weights, hidden_weights, input_bias, hidden_bias, m , L, alpha):
    '''
    Basic funciton to test the average errors in the ANN with both positive and negative test data
    
    Inputs: positive & negative test data (as 34-bit vectors in rows; sample count must be the same), input & hidden weight matrices, input and bias weight matrices, m, L, alpha
    Outputs: total average positive and negative errors resulting from input data being run through network 
    '''
    pos_errors = [] # Store all accrued errors
    neg_errors = []
    
    # Iterate through all positive and negative test data
    for index in range(0,pos_data.shape[0]):
        pos_input = np.asmatrix(pos_data[index])
        neg_input = np.asmatrix(neg_data[index])
        
        # Run through neural network, get resulting error
        last_pos_error = NN_input(pos_input, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha, True, False)
        last_neg_error = NN_input(neg_input, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha, False, False)
        
        # Add errors to list
        pos_errors.append(last_pos_error)
        neg_errors.append(last_neg_error)
    
    # After all iterations are complete, calculate average errors
    avg_pos_error = sum(pos_errors) / float(len(pos_errors))
    avg_neg_error = sum(neg_errors) / float(len(neg_errors))
    
    return avg_pos_error, avg_neg_error
            
            
def NN_input(sample_input, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha, pos_sample = True, update = False):    
    '''
    Funtion to take sample input (one by one) and run them through the neural network as defined by the weight and bias matrices.
    If the update flag is True, then it will update these matrices through back propagation and return them; otherwise it simply returns the error.
    
    Input:  sample input (vector of 34 bits of 0,1), input & hidden matrices, weight & bias matricies, 
            total sample size, lambda, alpha for training purposes, a flag to define if desired output is 0 or 1, update flag
    Output: If updating – output error, updated input and hidden weight matrices, updated input and hidden bias matrices.
            If not updating – output error
    '''      
    if pos_sample: # Desired output of positive value = 1, negative value = 0
        desired_output = np.ones((1,1))
    else:
        desired_output = np.zeros((1,1))    
    
    # Given the sample input and weight matrix, tranform via the dot product, add bias matrix
    input_transform = np.dot(sample_input, input_weights) + input_bias
    
    # Pass through the activation function
    hidden_layer = sigmoid(input_transform)
    
    # Again, add bias and propagate predicted values through the weights between hidden + output layers
    hidden_transform = np.dot(hidden_layer, hidden_weights) + hidden_bias
    output = sigmoid(hidden_transform)

    # Get the difference between the desired output and the result from the autoencoder
    output_error = desired_output - output
    
    if update: # Update all weight and bias matrices and return these updates along with the error
        # Take the error weighted derivative to avoid changing weights that lead to post-sigmoid values close to either 0 or 1 
        # Because these are probably starting to converge -> focus effort on changing ambiguous (~ 0.5) values
        output_delta = output_error * dsigmoid(output)
    
        # Backpropagate to determine the error in the hidden layer
        hidden_layer_error = output_delta.dot(hidden_weights.T)
        
        # Another error-weighted derivative
        hidden_layer_delta = np.multiply(hidden_layer_error, dsigmoid(hidden_layer))
    
        # Update the weights  and bias matrices for each layer
        hidden_weights += alpha * (((1 / m ) * (hidden_layer.T.dot(output_delta))) + L * hidden_weights)
        hidden_bias += (alpha / m) * output_delta
        
        input_weights += alpha * (((1 / m ) * (sample_input.T.dot(hidden_layer_delta))) + L * input_weights)
        input_bias += (alpha / m) * hidden_layer_delta
        
        return np.mean(np.abs(output_error)), input_weights, hidden_weights, input_bias, hidden_bias
    
    return np.mean(np.abs(output_error))    # Not updating, just return the error
    

def get_neuralnet_predictions(samples, input_weights, hidden_weights, input_bias, hidden_bias):
    '''
    Given a pre-trained neural network, take a series of samples and provide predictions as to their Rap1 binding likelihood
    
    Inputs: sample data (as 34-bit vectors in rows), input & hidden weight matrices, input and bias weight matrices
    Outputs: total average positive and negative errors resulting from input data being run through network 
    '''
    output_list = [] # Store all predicted output values
    
    # Iterate through all test data
    for index in range(0,samples.shape[0]):
        sample_input = np.asmatrix(samples[index]) # Pick one row at a time
        
        # Given the sample input and weight matrix, tranform via the dot product, add bias matrix
        input_transform = np.dot(sample_input, input_weights) + input_bias
        
        # Pass through the activation function
        hidden_layer = sigmoid(input_transform)
        
        # Again, add bias and propagate predicted values through the weights between hidden + output layers
        hidden_transform = np.dot(hidden_layer, hidden_weights) + hidden_bias
        output = float(sigmoid(hidden_transform))
    
        # Get the difference between the desired output and the result from the autoencoder
        output_list.append(output)
        
    return output_list
    
    