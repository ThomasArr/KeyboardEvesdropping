import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio


class AudioUtil:
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, newsr

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------

    @staticmethod
    def pad_trunc(aud, n_before_peak, n_after_peak):
        sig, sr = aud
        peak_indice = 0
        peak_value = 0
        for i in range(len(sig[0])):
            if sig[0][i] > peak_value:
                peak_value = sig[0][i]
                peak_indice = i
        return sig[:, peak_indice - n_before_peak:peak_indice + n_after_peak], sr

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec
