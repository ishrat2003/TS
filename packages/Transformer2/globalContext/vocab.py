import os, sys
import lc
import operator
import utility
import numpy
from .store import Store
import datetime
import json

class Vocab(Store):

    def __init__(self, dataset, path):
        super().__init__(dataset, path)
        self.vocab = {}
        self.index = 0
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.processedSentences = []
        self.sequenceIndex = 0
        return


    def setDatasetPath(self, path):
        self.datasetPath = path
        return


    def getVocab(self):
        return self.vocab


    '''
    allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    '''
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return


    def buildVocab(self):
        self.vocab = {}
        for item in self.dataset:
            rawData, _ = item
            self.__processText(self.getText(rawData.numpy()))

        self.__sort()
        self._save()
        print('Total vocab: ', len(self.vocab))
        print('Total sentences: ', len(self.processedSentences))
        print('Finished processing')
        return


    def getText(self, rawData):
        data = json.loads(rawData.decode("utf-8"))
        abstractText = [paragraph["text"] for paragraph in data["abstract"]]
        bodyText = [paragraph["text"] for paragraph in data["body_text"]]
        text = data["metadata"]["title"] + '. ' + ' '.join(abstractText) + ' '.join(bodyText)
        return { 'text': text}


    def _load(self):
        self._loadVocab()
        self._loadSentences()
        return


    def _save(self):
        self._saveNumpy('vocab.npz', list(self.vocab.values()))
        # self._saveNumpy('sentences.npz', list(self.processedSentences))
        return


    def _loadVocab(self):
        if not hasattr(self, 'vocab'):
            self.vocab = {}
        
        vocabFiles = self._loadNumpy('vocab.npz')

        if vocabFiles is not None:
            for fileRef in vocabFiles:
                vocabData = vocabFiles[fileRef]
                if vocabData is not None:
                    for word in vocabData:
                        stemmedWord = word['stemmed_word']
                        self.vocab[stemmedWord] = word

        return


    def _loadSentences(self):
        self.processedSentences = []
        for item in self.dataset:
            rawData, _ = item
            sentences = self.__processText(self.getText(rawData.numpy()))
            self.__addToSentence(sentences)
            
        # sentenceFiles = self._loadNumpy('sentences.npz')
        # self.processedSentences = []
        # if sentenceFiles is not None:
        #     for fileRef in sentenceFiles:
        #         sentenceData = sentenceFiles[fileRef]
        #         if sentenceData is not None:
        #             for sentence in sentenceData:
        #                 self.processedSentences.append(sentence)
        return


    def __processText(self, details):
        lcProcessor = self.__getLCProcessor(details)

        localWords = lcProcessor.getWordInfo()
        self.__addToVocab(localWords, details)

        return lcProcessor.getSentences()
    
    def __getLCProcessor(self, details):
        text = details['text']
        lcProcessor = lc.Peripheral(text, 0)
        lcProcessor.setAllowedPosTypes(self.allowedPOSTypes)
        lcProcessor.setPositionContributingFactor(1)
        lcProcessor.setOccuranceContributingFactor(1)
        lcProcessor.setProperNounContributingFactor(1)
        lcProcessor.setTopScorePercentage(0.2)
        lcProcessor.setFilterWords(0.2)
        lcProcessor.train()
        lcProcessor.loadFilteredWords()
        return lcProcessor


    def __addToSentence(self, sentences):
        if len(sentences) == 0:
            return

        currentKeys = self.vocab.keys()
        for sentence in sentences:
            processedSentence = []
            for word in sentence:
                if word not in currentKeys:
                    continue
                wordDetails = self.vocab[word]
                if 'sort_index' in wordDetails.keys():
                    processedSentence.append(wordDetails['sort_index'])
                    
            if len(processedSentence) > 1:
                self.processedSentences.append(processedSentence)
        return


    def __addToVocab(self, words, details):
        if 'timestamp' in details.keys():
            self.sequenceIndex = details['timestamp']
        else:
            # Counting lines
            self.sequenceIndex += 1 
        
        totalWords = len(words)
        if not words:
            return

        currentVocabLength = len(self.vocab)
        currentKeys = self.vocab.keys()
        for word in words.keys():
            currentWordKeys = words[word].keys()
            if word not in currentKeys:
                wordDetails = {}
                wordDetails['number_of_blocks'] = 1
                wordDetails['total_count'] = words[word]['count']
                wordDetails['label'] = words[word]['pure_word']
                wordDetails['stemmed_word'] = words[word]['stemmed_word']
            else:
                wordDetails = self.vocab[word]
            
            if 'score' in currentWordKeys:
                wordDetails['score'] = words[word]['score']
            else:
                wordDetails['score'] = 0
            
            if 'appeared' not in currentWordKeys:
                wordDetails['appeared'] = self.sequenceIndex
                wordDetails['index'] = currentVocabLength
                currentVocabLength += 1
            else:
                wordDetails = self.vocab[word]
                wordDetails['number_of_blocks'] += 1
                wordDetails['total_count'] += words[word]['count']
                
            if 'score' in currentWordKeys:
                wordDetails['score'] += words[word]['score']

            self.vocab[word] = wordDetails
        return


    def __sort(self, attribute = 'number_of_blocks'):
        if len(self.vocab) == 0:
            return
        print('vocab length before sorting: ', len(self.vocab))
        sortedVocab = {}
        index = 0
        for value in sorted(self.vocab.values(), key=operator.itemgetter(attribute), reverse=True):
            if value['total_count'] == 1:
                continue
            value['sort_index'] = index
            sortedVocab[value['stemmed_word']] = value
            index += 1
            print(value)

        self.vocab = sortedVocab
        print('vocab length before sorting: ', len(self.vocab))
        return


