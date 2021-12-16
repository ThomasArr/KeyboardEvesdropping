# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np

from AudioUtil import AudioUtil

# myds = SoundDS("./data/")
#
# data_length = len(myds)
# train_length = round(data_length * 0.8)
# test_length = data_length - train_length
#
# train_ds, val_ds = random_split(myds, [train_length, test_length])
from SoundDS import SoundDS

# from convolutionalNetwork import ConvNetwork
from MLPClassifier import MLPClassifier
from RFClassifier import RFClassifier


def t():
    audio = AudioUtil.open("./data/q1.wav")
    audio = AudioUtil.pad_trunc(audio)

    melspec = AudioUtil.spectro_gram(audio)

    # melspec_augmented = AudioUtil.spectro_augment(melspec)
    print(melspec.shape)
    print(AudioUtil.tensorToNumpy(melspec))
    # AudioUtil.plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')

    # plt.show()


DS = SoundDS(train_data_path="./data/", test_data_path="./data/test/")

# AudioUtil.plot_spectrogram(DS.data[1], title="MelSpectrogram - torchaudio", ylabel='mel freq')
# cnn_model = ConvNetwork(DS.data, DS.label)

model = MLPClassifier()
#print(model.crossValidate(DS.train_data, DS.train_label)) #[0.83333333 0.88888889 0.94117647 0.94117647 0.88235294]
#print(model.random_tuning(DS.train_data, DS.train_label))
model.train(DS.train_data, DS.train_label)
print(model.score(DS.train_data, DS.train_label))
print(model.predict(DS.test_data))
