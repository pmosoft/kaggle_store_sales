import numpy as np

def make_sequence_dataset(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i + window_size])
        label_list.append(label[i + window_size])

    return np.array(feature_list), np.array(label_list)
