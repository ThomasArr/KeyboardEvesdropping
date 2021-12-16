from AudioUtil import AudioUtil


class DataCreator:

    @staticmethod
    def createSamples(listPath, windowSize=7000, intensityThreshold=0.1, n_before_peak=400, n_after_peak=6600):
        sig, sr = AudioUtil.open(listPath)
        n = 0
        index = 0
        letter = DataCreator.getLetterFromPath(listPath)
        while index + 7000 < len(sig[0]):
            windowMax = 0
            windowMaxIndex = index
            for k in range(0, windowSize):
                if sig[0][index + k] > windowMax:
                    windowMax = sig[0][index + k]
                    windowMaxIndex = index + k
            if windowMax > intensityThreshold:
                path = "./data/test/" + str(letter) + str(n) + ".wav"
                waveform = sig[:, windowMaxIndex - n_before_peak:windowMaxIndex + n_after_peak]
                AudioUtil.save(path, (waveform, sr))
                n += 1
            index += windowSize

    @staticmethod
    def getLetterFromPath(path):
        t = path.split("/")
        return t[-1][0]


DataCreator.createSamples("./data/DataList/xxx2_list.wav")
