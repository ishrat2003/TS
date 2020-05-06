from .cwr import CWR
from .visualization import Visualization
import numpy
import utility
import math
from nltk.stem.porter import PorterStemmer
from .meta import Meta
import json, operator

class RelativeCWR(CWR):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        self.totalZones = 5
        self.totalFonts = 50
        self.stemmer = PorterStemmer()
        self.wordToProcess = None
        self.maxRadius = 0
        self.relatedDocs = []
        self.localContextWords = {}
        self.jsonData = {
            'positive_sentiment': [],
            'negative_sentiment': [],
            'same_topic': [],
            'core_context': [],
            'other_topic': []
        }
        self.relativeDistance = 0.8
        self.minNumberOfDocsAppeared = 5
        return
    
    
    def setRelativeDistance(self, distancePercentage):
        self.relativeDistance = distancePercentage / 100;
        return
    
    def setMaximumRadius(self):
        for wordIndex in self.localContextWords.keys():
            if self.maxRadius < self.localContextWords[wordIndex]:
                self.maxRadius = self.localContextWords[wordIndex] + 10
        return
    
    def getContext(self, word, minNumberOfDocsAppeared = 5):
        self.setWord(word)
        
        if not self.wordToProcess:
            return
        
        self.minNumberOfDocsAppeared = minNumberOfDocsAppeared
        self.setRelatedDocuments()
        self.setLocalContextWords()
        points = self.process(self.minNumberOfDocsAppeared)
        print('Total points: ', len(points))
        self.saveJson()
        return points
    
    def saveJson(self):
        children = []
        for category in self.jsonData.keys():
            children.append({
                'name': category,
                'size': len(self.jsonData[category]),
                'children': self.getChildrenFromPoints(self.jsonData[category])
            })
        
        graphData = {
            'name': self.wordToProcess['label'],
            'size': len(self.jsonData.keys()),
            'children': children
        }
        
        with open(self._getBubblePath(), 'w') as outfile:
            json.dump(graphData, outfile)
        return
    
    def getChildrenFromPoints(self, points):
        points = self.sortPoints(points)
        children = []
        for point in points:
            children.append({
                'name': point['label'],
                'size': point['relative_num_blocks']
            })
        return children
    
    def shouldInclude(self, word):
        wordIndex = word['index']
        if wordIndex not in self.localContextWords.keys():
            return False
        
        if self.localContextWords[wordIndex] <= self.minNumberOfDocsAppeared:
            return False
        
        return True
    
    
    def afterProcess(self, point):
        point['relative_num_blocks'] = self.localContextWords[point['index']]
        
        if not self.shouldInclude(point):
            return point
        
        
        wordRadius = self.getRadius(self.wordToProcess)
        allowedDistance = self.maxRadius * self.relativeDistance
        startCore = wordRadius - allowedDistance
        endCore = wordRadius
        
        
        if point['sentiment'] == 'positive':
            self.jsonData['positive_sentiment'].append(point)
        elif point['sentiment'] == 'negative':
            self.jsonData['negative_sentiment'].append(point)
        elif (point['number_of_blocks'] >= self.maxRadius):
            distance = math.fabs(wordRadius - point['radius'])
            if distance <= allowedDistance:
                self.jsonData['core_context'].append(point)
        else:
            distance = math.fabs(wordRadius - point['radius'])
            if distance <= allowedDistance:
                if self.wordToProcess['topic'] == point['topic']:
                    self.jsonData['same_topic'].append(point);
                else:
                    self.jsonData['other_topic'].append(point);
        
        return point
    
    
    def getRadius(self, word):
        return self.maxRadius - self.localContextWords[word['index']]
    
    def setLocalContextWords(self):
        if not self.relatedDocs:
            return
        
        meta = Meta(self.datasetProcessor)
        docLcs = meta.loadDocLcs()
        
        self.localContextWords = {}
        
        for doc in self.relatedDocs:
            if doc not in docLcs.keys():
                continue
            
            lcWords = docLcs[doc]
            self.appendLcWords(lcWords)
            
        del docLcs
        del self.relatedDocs
        
        print('Total Lc words: ', len(self.localContextWords))
        return
    
    def appendLcWords(self, lcWords):
        for wordIndex in lcWords:
            if wordIndex not in self.localContextWords.keys():
                self.localContextWords[wordIndex] = 0
                
            self.localContextWords[wordIndex] += 1
        return
    
    def setRelatedDocuments(self):
        meta = Meta(self.datasetProcessor)
        wordDocs = meta.loadWordDocs()
        
        if self.wordToProcess['index'] not in wordDocs.keys():
            print('Related documents not found.')
            return 
        
        self.relatedDocs = wordDocs[self.wordToProcess['index']]
        print('Total docs: ', self.maxRadius)
        return
    
    def setWord(self, word):
        word = self.stemmer.stem(word.lower())
        
        if word not in self.vocab.keys():
            print('Failed to find the word.')
            return
        
        self.wordToProcess = self.vocab[word]
        self.prefix = word + '_'
        return
    
    
    def sortPoints(self, points, attribute = 'relative_num_blocks'):
        if not len(points):
            return []
        
        sortedPoints = []
        
        for value in sorted(points, key=operator.itemgetter(attribute), reverse=True):
            sortedPoints.append(value)
        print('sorted: ', sortedPoints)
        return sortedPoints
    
    
    def _getBubblePath(self):
        	return self._getFilePath(self.prefix + '_bubble.json', self.path)
    