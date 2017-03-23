import numpy as np
import copy

def read_pos_data(filename):
    '''
    IO function to read in positive data, given in the format of 17 BP sequences.
    
    Input: filename of .txt file with 17 BP character sequences separated by newlines
    Output: list of sequences as characters
    '''
    sequences = []
    with open(filename,'r') as f:
        for line in f:
            seq = line.strip()
            sequences.append(seq)
    return sequences


def read_neg_data(filename):
    '''
    IO Function to read in negative data in the form of .fa files
    
    Input: filename of .fa file with FASTA formatted dna sequences
    Output: list of sequences as character strings
    '''
    sequences = []
    with open(filename,'r') as f:
        
        while True:
            line = f.readline()
            if line == '': break
            if(line.startswith(">")): # Record sequence
                line = f.readline()
                if(line.startswith(">")): # Brute check
                    print('if you\'re reading this we broke something')
                    print(line)
                    break
                k = 0
                seq = ''
                while(k < 16): # I hate reading in files
                    seq += line.strip()
                    line = f.readline()
                    k += 1
                sequences.append(seq)
    return sequences

def get_short_seqs(sequences, length, seed_id = np.random.randint(0,100)):
    '''
    IO function to take long ( > 17 bp) sequences and pick a random 17 sequential bp subset
    
    Input: list of sequences as character strings, length of desired sequence, (random seed)
    Output: list of truncated sequences as character strings
    '''
    np.random.seed(seed_id) # For testing purposes
    short_seqs = []
    for seq in sequences:
        start = np.random.randint(0,len(seq) - length)
        short_seqs.append(seq[start:start + length])
    return short_seqs


def seq_to_bit_vector(sequences):
    '''
    IO function to take in a list of sequences as character strings and transform them to bit vectors representing the sequencial characters
    Each base pair corresponds to a 2 digit bit as follows: {'A':[0, 0], 'C':[0, 1], 'G':[1, 0], 'T':[1, 1]}
    
    Input: list of sequences as character strings
    Output: numpy matrix of 'bit vectors' (0's and 1's), 34 columns in length. Rows represent individual sequences
    '''
    bit_sequences = np.empty([1,34], dtype = float)
    bit_dict = {'A':[0, 0], 'C':[0, 1], 'G':[1, 0], 'T':[1, 1]}
    for seq in sequences:
        bit_seq = np.empty([1,34], dtype = float)
        for index, char in enumerate(seq):
            bit_seq[0][index * 2] = float(bit_dict[char][0])
            bit_seq[0][index * 2 + 1] = float(bit_dict[char][1])        
        bit_sequences = np.append(bit_sequences, bit_seq, axis = 0)
    bit_sequences = np.delete(bit_sequences,0,0) # I know there's a better way to do this... but what is it?
    return bit_sequences

def sample_neg_data(sequences, num, seed_id = np.random.randint(0,100)):
    '''
    IO function to take a random subet of sequences given a list (used in this case for the large negative dataset)
    
    Input: list of sequences as character strings, number of desired sequences, (random seed)
    Output: truncated list of sequences as character strings
    '''
    np.random.seed(seed_id) # For testing purposes
    sample_seqs = []
    random_seqs = np.random.randint(0,len(sequences), size = num)
    for index in random_seqs:
        sample_seqs.append(sequences[index])
    return sample_seqs

def split_kfold_groups(pos_data_groups, neg_data_groups, groups, split_index):
    '''
    IO function to split equal sizes of positive and negative data into subgroups of size 1 (for testing) and k-1 (for training)
    
    Input: positive and negative data as numpy matrix of bits, number of desired groups, index of group to select as test group
    Output: positive and negative data as numpy matricies of bits in k-fold groups for testing (size 1) and training (size k-1)
    '''
    group_index = list(range(0, groups))
    pos_data_copy = copy.deepcopy(pos_data_groups)
    neg_data_copy = copy.deepcopy(neg_data_groups)
    
    test_index = group_index[split_index]
    
    test_pos_data = np.asmatrix(pos_data_copy[test_index])
    test_neg_data = np.asmatrix(neg_data_copy[test_index])
    
    del pos_data_copy[test_index]
    del neg_data_copy[test_index]   
    
    train_pos_data = np.empty([1,34], dtype = float)
    train_neg_data = np.empty([1,34], dtype = float)    

    
    for j in range(0, groups - 1): # iterate through remaining groups in data_copy lists
        for index in range(0, pos_data_copy[j].shape[0]): # iterate through each sequence in remaining groups
            pos_group = np.asmatrix(pos_data_copy[j])
            neg_group = np.asmatrix(neg_data_copy[j])
            
            train_pos_data = np.append(train_pos_data, pos_group[index], axis = 0)
            train_neg_data = np.append(train_neg_data, neg_group[index], axis = 0)
            
    train_pos_data = np.delete(train_pos_data, 0, 0)
    train_neg_data = np.delete(train_neg_data, 0, 0)
    
    return test_pos_data, test_neg_data, train_pos_data, train_neg_data

