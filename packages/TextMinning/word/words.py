from .word import Word
import utility
import re
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer

class Words(Word):
    
    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        return
    
    
    def appendWords(self, words):
        if not len(words):
            return ''
        ids = []
        for stemmedWord in words.keys():
            index = self.appendWord(stemmedWord, words[stemmedWord])
            if index:
                ids.append(index)
            
        return ids
    