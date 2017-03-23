import numpy as np
from bmi203fp import io_tools, methods

def test_autoencoder():
    
    # Set up initial run parameters for autoencoder
    input_layer_size = 8
    hidden_layer_size = 3
    output_layer_size = 8
    L = 0
    alpha = 50
    iterations = 10000
    max_tolerated_error = 0.01
    
    # Train autoencoder, ensure functionality
    input_data = np.identity(8)
    desired_output = input_data
    
    final_layer, final_error = methods.train_autoencoder(input_data, desired_output, input_layer_size, hidden_layer_size, output_layer_size, L, alpha, iterations)

    assert final_error < max_tolerated_error
    
def test_neuralnetwork():
    # Set up run parameters to test-train neural net
    input_layer_size = 34
    hidden_layer_size = 16
    L = 0
    alpha = 50
    iterations = 100
    convergence_criteria = 0.0001
    max_tolerated_error = 0.1
    
    pos_data = io_tools.read_pos_data('rap1-lieb-positives.txt')
    neg_data = io_tools.read_neg_data('yeast-upstream-1k-negative.fa')
    
    # Clean up data subset for training purposes
    train_pos_data = pos_data[0:136]
    train_neg_data = io_tools.sample_neg_data(neg_data, 136) # Randomly sample negative data
    train_neg_data = io_tools.get_short_seqs(train_neg_data, 17) # Convert to 17 bp sequences
    
    # Convert data to 34-bit vectors and place in matrices
    all_pos_data = io_tools.seq_to_bit_vector(train_pos_data)
    all_neg_data = io_tools.seq_to_bit_vector(train_neg_data)
    
    # Test-train neural network
    avg_pos_error, avg_neg_error, input_weights, hidden_weights, input_bias, hidden_bias = methods.train_neuralnet(all_pos_data, all_neg_data, input_layer_size, hidden_layer_size, L, alpha, iterations, convergence_criteria, True)
    
    avg_total_error = (avg_pos_error + avg_neg_error) / 2
    
    assert avg_total_error < max_tolerated_error
        
        
        
        
        