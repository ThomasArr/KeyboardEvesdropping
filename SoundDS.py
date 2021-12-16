from torch.utils.data import DataLoader, Dataset, random_split
import os
# ----------------------------
# Sound Dataset
# ----------------------------
from AudioUtil import AudioUtil
import numpy as np


def reduce_data_size(x, width=2, length=4):
    new_shape = x.shape[0] // length, x.shape[1] // width
    new_x = np.zeros((new_shape[0], new_shape[1]))

    for i in range(0, new_shape[0]):
        for j in range(0, new_shape[1]):
            new_x[i, j] = np.mean(x[i * length:(i + 1) * length, j * width:(j + 1) * width])
    return new_x


class SoundDS(Dataset):
    def __init__(self, train_data_path, test_data_path):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        SoundDS.createTrainData(self, train_data_path)
        SoundDS.createTestData(self, test_data_path)

    def createDS(self, path):
        data = []
        label = []
        for file in os.listdir(path):
            if file.endswith(".wav"):
                temp = reduce_data_size(AudioUtil.soundToNumpySpectrogram(audio_file=path + file))
                data.append(temp.flatten())
                label.append(file[0])
        return np.array(data), np.array(label)

    def createTestData(self, path):
        data, label = self.createDS(path)
        self.test_data = data
        self.test_label = label

    def createTrainData(self, path):
        data, label = self.createDS(path)
        self.train_data = data
        self.train_label = label
