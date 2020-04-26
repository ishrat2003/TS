
import lc as LC
from .words import Words
import operator

class Meta(Words):
    
    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.metaFile = self.getFile('meta')
        self.docsLCFile = []
        return
    
    def process(self):
        data = self.datasetProcessor.getTrainingSet()

        for item in data:
            sourceText = self.datasetProcessor.getText(item)
            words = self.getLC(sourceText)
            indices = self.appendWords(words)
            self.docsLCFile.append(indices)

        self.sortVocab() 
        for word in self.vocab.keys():
            self.metaFile.write(self.vocab[word])
            
        
        self.saveVocab()
        self.saveDocLcs()
        return

    def getLC(self, text):
        self.lc = LC.Peripheral('')
        self.lc.setAllowedPosTypes(['NN', 'NNP', 'NNS', 'NNPS'])
        self.lc.setPositionContributingFactor(5)
        self.lc.setOccuranceContributingFactor(0)
        self.lc.setProperNounContributingFactor(0)
        self.lc.setTopScorePercentage(0.2)
        self.lc.setFilterWords(0)

        self.lc.loadSentences(text)
        self.lc.loadFilteredWords()
        self.lc.train()
        self.lc.getPoints() # Doing the training
        words = self.lc.getFilteredWords()
        contributors = self.lc.getContrinutors()
        
        if not len(words) or not len(contributors):
            return {}
        
        processedWords = {}
        
        for item in contributors:
            if item in words.keys():
                processedWords[item] = words[item]
                
        return processedWords
    
    def saveDocLcs(self):
        self._saveInPickel(self._getDocLcsPath(), self.docsLCFile)
        return
    
    def loadDocLcs(self):
        self.docsLCFile = self._getFromPickel(self._getDocLcsPath())
        return self.docsLCFile

    def _getDocLcsPath(self):
    	return self._getFilePath('docs_lcs.sav', self.path)
