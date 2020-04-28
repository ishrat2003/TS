from .meta import Meta
from .visualization import Visualization
import numpy
import utility
import json

class Hierarchy(Visualization):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        self.data = {}
        
        meta = Meta(self.datasetProcessor)
        self.docsLCs = meta.loadDocLcs()
        return

    def process(self, minNumberOfDocsAppeared = 5):
        if not len(self.vocab):
            return
        self.remove()
        words = self.setFilteredWordsByTopics(minNumberOfDocsAppeared)
        self.data = self.prepareData()
        self.save()
        return self.data
    
    def prepareData(self, zones = 5):
        self.wordCoOccurance = self.getCoOccurrence()
        self.data = {
            "name": "cwr"
        }
        children = []
        for topic in self.filteredWordsByTopic.keys():
            self.setMinMax(self.filteredWordsByTopic[topic])
            self.gap = (self.max - self.min) / zones
            start = self.max - self.gap 
            end = self.max
            self.loadTopicPoints(self.filteredWordsByTopic[topic])
            childrenItems = self.getSiblings(self.filteredWordsByTopic[topic], start, end, 1)
            topicData = {
                "name": topic,
                "children": childrenItems
            }
            children.append(topicData)
        
        self.data["children"] = children
        return self.data
    
    def setMinMax(self, points):
        self.min = 99999
        self.max = 0
        
        for point in points:
            if point['number_of_blocks'] > self.max:
                self.max = point['number_of_blocks']
            if point['number_of_blocks'] < self.min:
                self.min = point['number_of_blocks']
        return
    
    def loadTopicPoints(self, points):
        self.indexedPoints = {}
        for point in points:
            self.indexedPoints[point['index']] = point
            
        return
    
    def getSiblings(self, points, start, end, level):
        siblings = []
        for point in points:
            if (point['number_of_blocks'] >= start) and  (point['number_of_blocks'] < end):
                item = {
                    "name": point['label'],
                    "size": point['number_of_blocks'],
                    "children": self.getChild(point['index'], end - self.gap)
                }
                siblings.append(item)
            
        return siblings
    
    def getChild(self, parentIndex, end):
        if parentIndex not in self.wordCoOccurance:
            return []
        
        prospectiveChildren = self.wordCoOccurance[parentIndex]
        start = end - self.gap 
        children = []
        
        for index in prospectiveChildren:
            if index == parentIndex:
                continue
            if index not in self.indexedPoints.keys():
                continue
            point = self.indexedPoints[index]
            if (point['number_of_blocks'] >= start) and (point['number_of_blocks'] < end):
                item = {
                    "name": point['label'],
                    "size": point['number_of_blocks'],
                    "children": self.getChild(point['index'], end - self.gap)
                }
                children.append(item)
        
        return children
        
    
    def getCoOccurrence(self):
        wordCoOccurance = {}
        for doc in self.docsLCs:
            for wordIndex1 in  doc:
                if wordIndex1 not in wordCoOccurance.keys():
                    wordCoOccurance[wordIndex1] = []
                for wordIndex2 in  doc:
                    wordCoOccurance[wordIndex1].append(wordIndex2)
                    
        return wordCoOccurance
    
    def save(self):
        with open(self._getPath(), 'w') as outfile:
            json.dump(self.data, outfile)
        return
    
    def remove(self):
        file = utility.File(self._getPath())
        file.remove()
        return
 
    def _getPath(self):
    	return self._getFilePath('hierarchy.json', self.path)

