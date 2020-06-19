import numpy
import utility
import math
from nltk.stem.porter import PorterStemmer
from .meta import Meta
import json, operator

class Relative():
    
    def __init__(self, datasetProcessor, params):
        self.datasetToProcess = datasetToProcess
        self.params = params
        self.wordToProcess = None
        self.stemmer = PorterStemmer()
        return
    
    
    def getContext(self, word, minNumberOfDocsAppeared = 5):
        self.wordToProcess = self.stemmer.stem(word.lower())
        self.minNumberOfDocsAppeared = minNumberOfDocsAppeared
        self.setRelatedDocuments()
        return
    
    def setRelatedDocuments(self):
        meta = Meta(self.datasetProcessor)
        words = meta.loadSummaryVocab()
        
        if self.wordToProcess not in words.keys():
            print('Related documents not found.')
            return 
        
        details = words[self.wordToProcess]
        print('Total docs: ', details)
        return
    