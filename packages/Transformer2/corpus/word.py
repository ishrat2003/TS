import utility
from topic.lda import LDA
import os
import pickle
import operator
from .writer import Writer

class Word(Writer):
    
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
    
    def appendWord(self, stemmedWord, details):
        if stemmedWord not in self.vocab.keys():
            self.vocab[stemmedWord] = {
                'index': len(self.vocab) + 1,
                'label': details['pure_word'],
                'count': details['count'],
                'number_of_blocks': 1,
                'topic': self.getTopic(details['pure_word']),
                'sentiment': self.getSentiment(details['pure_word']),
                'stemmed_word': stemmedWord
            }
            return self.vocab[stemmedWord]['index']
        
        self.vocab[stemmedWord]['number_of_blocks'] += 1
        self.vocab[stemmedWord]['count'] += details['count']
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
    
    def _getVocabPath(self):
    	return self._getFilePath('meta_vocab.sav', self.path)

