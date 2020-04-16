import os
import tensorflow_datasets as tfds
import utility
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer

class Base:
    
    def __init__(self, path, name = None, supervised = True):
        self.directoryPath = path
        self.name = name
        self.totalItems = 0
        self.splitPercentage = 100
        self.metadata = None
        self.supervised = supervised
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.stopWords = utility.Utility.getStopWords()
        
        if name:
            self.path = os.path.join(path, self.name)
        else:
            self.path = path
        return
    
    def setSplitPercentage(self, percentage = 100):
        if ((percentage < 0) or (percentage > 100)):
            percentage = 100
        self.splitPercentage = percentage
        return
    
    def setTotalItems(self, total = 100):
        self.totalItems = total
        return
    
    def get(self):
        self.dlConfig = tfds.download.DownloadConfig(manual_dir=self.directoryPath)
        return self.dataset()
    
    def getTrainingSet(self):
        trainingSet, validationSet = self.get()
        return trainingSet
    
    def getMetadata(self):
        return self.metadata
    
    def getProcessedPath(self):
        return self.path
    
    def getVocabPath(self):
        return os.path.join(self.path, "vocab")

    def dataset(self):
        return None, None
    
    def getGenerator(self, trainingSet, type = 'source'):
        return None
    
    def getText(self, rawData):
        text, summary = rawData
        return text.numpy().decode('utf-8')
    
    def getAbstract(self, rawData):
        text, summary = rawData
        return summary.numpy().decode('utf-8')
    
    def getTitle(self, rawData):
        return self.getAbstract(rawData)
    
    def getLabel(self, rawData):
        return 'Todo'
    
    def getAbstract(self, rawData):
        text, summary = rawData
        return summary.numpy()
    
    def getProcessedText(self, text):
        words = word_tokenize(self.__clean(text))
        allWords = pos_tag(words)
        processedWords = []

        for itemWord in allWords:
            (word, type) = itemWord
            if (type not in self.allowedPOSTypes) or (word in self.stopWords):
                continue

            word = self.__cleanWord(word)
            word = self.stemmer.stem(word.lower())
            processedWords.append(word)

        return ' '.join(processedWords)
    
    def _load(self, key): 
        if self.totalItems:
            readInstructions = [
                tfds.core.ReadInstruction('train', from_ = 1, to = self.totalItems + 1, unit='abs'),
                tfds.core.ReadInstruction('validation', from_ = 1, to = self.totalItems + 1, unit='abs'),
            ]
        elif self.splitPercentage:
            readInstructions = [
                tfds.core.ReadInstruction('train', to = self.splitPercentage, unit='%'),
                tfds.core.ReadInstruction('validation', to = self.splitPercentage, unit='%'),
            ]
        else:
            readInstructions = ['train', 'validation']  
            
        data, self.metadata = tfds.load(key, 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = self.supervised, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig),
            split = readInstructions)
        
        return data[0], data[1]
    