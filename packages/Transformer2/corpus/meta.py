
import lc as LC
from .words import Words
import operator
import utility

class Meta(Words):
    
    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.metaFile = self.getFile('meta')
        self.docsLCFile = {}
        self.wordDocs = {}
        return
    
    def process(self):
        data = self.datasetProcessor.getTrainingSet()

        for item in data:
            sourceText = self.datasetProcessor.getText(item)
            words = self.getLC(sourceText)
            indices = self.appendWords(words)
            label = self.datasetProcessor.getLabel(item)
            self.docsLCFile[label] = indices
            
            for index in indices:
                if index not in self.wordDocs.keys():
                    self.wordDocs[index] = []
                if label not in self.wordDocs[index]:
                    self.wordDocs[index].append(label)

        self.saveWordDocs()
        del self.wordDocs
        
        self.saveDocLcs()
        del self.docsLCFile
        
        self.sortVocab() 
        for word in self.vocab.keys():
            self.metaFile.write(self.vocab[word])
            
        self.saveVocab()
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
    
    def saveWordDocs(self):
        self._saveInPickel(self._getWordDocPath(), self.wordDocs)
        return
    
    def loadWordDocs(self):
        self.wordDocs = self._getFromPickel(self._getWordDocPath())
        return self.wordDocs
    
    def remove(self):
        super().remove()
        file = utility.File(self._getDocLcsPath())
        file.remove()
        csvFile = self.getFile('cwr_gc.csv')
        csvFile.remove()
        self.metaFile.remove()
        return

    def _getDocLcsPath(self):
    	return self._getFilePath('docs_lcs.sav', self.path)

    def _getWordDocPath(self):
        	return self._getFilePath('word_docs.sav', self.path)
