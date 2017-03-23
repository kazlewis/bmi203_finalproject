import io_tools
import methods

import numpy as np

# Set up initial run parameters for autoencoder
input_layer_size = 8
hidden_layer_size = 3
output_layer_size = 8
L = 0
alpha = 50
iterations = 100000

# Train autoencoder, ensure functionality
input_data = np.identity(8)
desired_output = input_data

#final_layer, final_error = methods.train_autoencoder(input_data, desired_output, input_layer_size, hidden_layer_size, output_layer_size, L, alpha, iterations)

# Set up run parameters to test-train neural net
input_layer_size = 34
hidden_layer_size = 16
L = 0
alpha = 50
iterations = 1000
convergence_criteria = 0.0001

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

avg_error = 1.0
ideal_hls = 1

# Cross validation – use k-fold validation with 8 groups (17 samples each)
groups = 8
pos_data_groups = np.vsplit(all_pos_data, groups)
neg_data_groups = np.vsplit(all_neg_data, groups)
m = 1 # dummy value for testing purposes

# Vector for ease of accounting of index of group splitting
group_index = list(range(0, groups))

# Iterate to find ideal hidden layer size
for hidden_layer_size in range(1,18):
    print('Testing HLS of ' + str(hidden_layer_size))

    # Keep track of results of validation
    kfold_results = []
    
    for split_index in group_index:
        print('Running k-fold validation with ' + str(groups) + ' groups, split index ' + str(split_index))
        # Split data in k-fold groups
        test_pos_data, test_neg_data, train_pos_data, train_neg_data = io_tools.split_kfold_groups(pos_data_groups, neg_data_groups, groups, split_index)
        
        # Train k-fold groups
        avg_pos_error, avg_neg_error, input_weights, hidden_weights, input_bias, hidden_bias = methods.train_neuralnet(train_pos_data, train_neg_data, input_layer_size, hidden_layer_size, L, alpha, iterations, convergence_criteria)
        
        # Test k-fold groups
        avg_pos_error, avg_neg_error = methods.test_neuralnet(test_pos_data, test_neg_data, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha)
        
        # Update results vector / list
        kfold_results.append((split_index, avg_pos_error, avg_neg_error))
        print('\tAvg Pos Error: ' + str(avg_pos_error) + '\n\tAvg Neg Error: ' + str(avg_neg_error) + '\n')
        
    total_avg_pos_error = 0.0
    total_avg_neg_error = 0.0
    
    for output in kfold_results:
        total_avg_pos_error += output[1]
        total_avg_neg_error += output[2]
        
    total_avg_pos_error = total_avg_pos_error / len(kfold_results)
    total_avg_neg_error = total_avg_neg_error / len(kfold_results)
    total_avg_error = (total_avg_pos_error + total_avg_neg_error) / 2
    
    print('Total Average Error = ' + str(total_avg_error))
    
    if total_avg_error < avg_error:
        ideal_hls = hidden_layer_size
        print('Found new ideal HLS: ' + str(ideal_hls))
        avg_error = total_avg_error

# Keep track of results of validation
kfold_results = []

for split_index in group_index:
    print('Running k-fold validation with ' + str(groups) + ' groups, split index ' + str(split_index))
    # Split data in k-fold groups
    test_pos_data, test_neg_data, train_pos_data, train_neg_data = io_tools.split_kfold_groups(pos_data_groups, neg_data_groups, groups, split_index)
    
    # Train k-fold groups
    avg_pos_error, avg_neg_error, input_weights, hidden_weights, input_bias, hidden_bias = methods.train_neuralnet(train_pos_data, train_neg_data, input_layer_size, hidden_layer_size, L, alpha, iterations, convergence_criteria)
    
    # Test k-fold groups
    avg_pos_error, avg_neg_error = methods.test_neuralnet(test_pos_data, test_neg_data, input_weights, hidden_weights, input_bias, hidden_bias, m, L, alpha)
    
    # Update results vector / list
    kfold_results.append((split_index, avg_pos_error, avg_neg_error))
    print('\tAvg Pos Error: ' + str(avg_pos_error) + '\n\tAvg Neg Error: ' + str(avg_neg_error) + '\n')
    
total_avg_pos_error = 0.0
total_avg_neg_error = 0.0

for output in kfold_results:
    total_avg_pos_error += output[1]
    total_avg_neg_error += output[2]
    
total_avg_pos_error = total_avg_pos_error / len(kfold_results)
total_avg_neg_error = total_avg_neg_error / len(kfold_results)
total_avg_error = (total_avg_pos_error + total_avg_neg_error) / 2

print('Total Average Error = ' + str(total_avg_error) + '\n')


# Test dataset from rap1-lieb-test.txt
test_data = io_tools.read_pos_data('rap1-lieb-test.txt')
test_data_input = io_tools.seq_to_bit_vector(test_data)
test_data_output = methods.get_neuralnet_predictions(test_data_input, input_weights, hidden_weights, input_bias, hidden_bias)

# Print final output
for index, seq in enumerate(test_data):    
    print(str(seq) + '\t' + str(test_data_output[index]) + '\t', end = '')















