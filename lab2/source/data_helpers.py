import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(politics_data_file, business_data_file, entertainment_data_file, sports_data_file, tech_data_file ):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    politics_examples = list(open(politics_data_file, "r").readlines())
    politics_examples = [s.strip() for s in politics_examples]
    business_examples = list(open(business_data_file, "r").readlines())
    business_examples = [s.strip() for s in business_examples]
    entertainment_examples = list(open(entertainment_data_file, "r").readlines())
    entertainment_examples = [s.strip() for s in entertainment_examples]
    sports_examples = list(open(sports_data_file, "r").readlines())
    sports_examples = [s.strip() for s in sports_examples]
    tech_examples = list(open(tech_data_file, "r").readlines())
    tech_examples = [s.strip() for s in tech_examples]
    # Split by words
    x_text = politics_examples + business_examples+entertainment_examples + sports_examples+tech_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    politics_labels = [[0,0, 1] for _ in politics_examples]
    business_labels = [[0,1, 0] for _ in business_examples]
    entertainment_labels = [[0,0, 1] for _ in entertainment_examples]
    sports_labels = [[0,1, 1] for _ in sports_examples]
    tech_labels = [[1,0, 1] for _ in tech_examples]


    y = np.concatenate([politics_labels, business_labels, entertainment_labels, sports_labels, tech_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
