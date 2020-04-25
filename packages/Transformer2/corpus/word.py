import utility
from topic.lda import LDA
import os
import pickle
import operator

class Word():
    
    def __init__(self, datasetProcessor):
        self.datasetProcessor = datasetProcessor
        self.path = self.datasetProcessor.getPath()
        self.stopWords = utility.Utility.getStopWords()
        self.positiveWords = utility.Utility.getPositiveWords()
        self.negativeWords = utility.Utility.getNegativeWords()
        lda = LDA(self.datasetProcessor, self.path)
        self.topics = lda.getTopics()
        self.vocab = {}
        return
    
    def getVocab(self):
        return self.vocab
    
    def appendWord(self, stemmedWord, label):
        if stemmedWord not in self.vocab.keys():
            self.vocab[stemmedWord] = {
                'index': len(self.vocab) + 1,
                'label': label,
                'number_of_blocks': 1,
                'topic': self.getTopic(label),
                'sentiment': self.getSentiment(label),
                'stemmed_word': stemmedWord
            }
            return self.vocab[stemmedWord]['index']
        
        self.vocab[stemmedWord]['number_of_blocks'] += 1
        return self.vocab[stemmedWord]['index']
    
    def getTopic(self, label):
        for topicIndex in self.topics:
            if label in self.topics[topicIndex]:
                return 'Topic ' + str(topicIndex)
        
        return "Others"
    
    def getSentiment(self, label):
        if label in self.positiveWords:
            return 'positive'
        if label in self.negativeWords:
            return 'negative'
        return 'normal'
    
    def sortVocab(self, attribute = 'number_of_blocks'):
        if not len(self.vocab):
            return
        sortedVocab = {}
        
        for value in sorted(self.vocab.values(), key=operator.itemgetter(attribute), reverse=True):
            sortedVocab[value['stemmed_word']] = value

        self.vocab = sortedVocab
        return
    
    def saveVocab(self):
        self._saveInPickel(self._getVocabPath(), self.vocab)
        return
    
    def loadVocab(self):
        self.vocab = self._getFromPickel(self._getVocabPath())
        return self.vocab
    
    def getFile(self, prefix = ''):
        path = os.path.join(self.path, prefix + 'vocab.csv')
        file = utility.File(path)
        return file
    
    def _getFilePath(self, fileName, path):
        return utility.File.join(path, fileName)
    
    def _getVocabPath(self):
    	return self._getFilePath('meta_vocab.sav', self.path)

    def _getFromPickel(self, filePath):
        file = utility.File(filePath)
        if file.exists():
            return pickle.load(open(filePath, 'rb'));
        return None

    def _saveInPickel(self, filePath, model):
        file = utility.File(filePath)
        if file.exists():
            file.remove()
        pickle.dump(model, open(filePath, 'wb'))
        return
