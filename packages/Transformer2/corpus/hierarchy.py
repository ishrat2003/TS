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
        self.wordCoOccurance = self.getCoOccurrence()
        self.totalZones = 5
        self.max = {}
        self.max['points'] = 0
        self.points = {}
        self.topics = []
        self.topicWordIds = {}
        self.processedIndexes = []
        self.data = []
        return
    
    def setTotalZones(self, totalZones):
        self.totalZones = totalZones
        return

    def process(self, minNumberOfDocsAppeared = 5):
        if not len(self.vocab):
            return
        
        self.remove()
        self.preparePoints()
        
        self.data = {'name': 'Hierarchy'}
        self.data['children'] = self.getTopics()
        self.data['size'] = len(self.data['children'])
        
        self.save()
        print(self.data)
        return self.data
    
    def getTopics(self):
        topics = []
        for topic in self.topics:
            topicItem = {}
            topicItem['name'] = topic
            topicItem['size'] = len(self.topicWordIds[topic])
            topicItem['children'] = self.getTopicItem(self.topicWordIds[topic], self.max[topic], topic)
            topics.append(topicItem)
        
        return topics
    
    def getTopicItem(self, wordIds, max, topic):
        children = []
        self.wordIdsByZones = self.getWordsByZones(wordIds, max)
        
        if not len(self.wordIdsByZones):
            return []

        for wordId in self.wordIdsByZones[0]:
            if wordId in self.processedIndexes:
                continue
            self.processedIndexes.append(wordId)
            item = {}
            item['name'] = self.points[wordId]['label']
            item['children'] = self.getChildren(wordId, 1, topic)
            item['size'] = len(item['children'])
            children.append(item)
            
            
        return children
    
    def getChildren(self, wordId, zone, topic):
        children = []
        candidates = []

        if zone > 25:
            return []

        if zone in self.wordIdsByZones.keys():
            candidates = self.wordIdsByZones[zone]
            if not len(candidates):
                return self.getChildren(wordId, zone+1, topic)
        elif (wordId in self.wordCoOccurance.keys()):
            candidates = self.wordCoOccurance[wordId]

        if not len(candidates):
            return children

        #common = list(set(candidates).intersection(relatedWords))
        
        #if not len(common):
        #    common = relatedWords
        # print(candidates)
        for itemId in candidates:
            point = self.points[itemId]
            #print('--topic -- ', topic, ' -- ', point['topic'])
            if (itemId == wordId) or (topic != point['topic']) or (itemId in self.processedIndexes):
              continue
            self.processedIndexes.append(itemId)
            item = {}
            item['name'] = self.points[itemId]['label']
            itemChildren = self.getChildren(itemId, zone+1, topic)
            item['size'] = len(itemChildren)
            item['children'] = itemChildren
            children.append(item)
        
        return children
    
    
    def getWordsByZones(self, wordIds, max):
        gap = max / self.totalZones
        zones = {}
        end = max
        start = end - gap
        zoneIndex = 0
        while start >= 0:
            zone = []
            for index in wordIds:
                point = self.points[index]
                if (point['number_of_blocks'] > start) and (point['number_of_blocks'] <= end):
                    zone.append(point['index'])
                    self.points[index]['zone'] = zoneIndex
                    
            zones[zoneIndex] = zone
            end = start
            start = end - gap
            zoneIndex += 1
        
        return zones
    
    def preparePoints(self):
        for word in self.vocab.keys():
            details = self.vocab[word]
            topic = details['topic']
            if topic not in self.topics:
                self.max[topic] = 0
                self.topicWordIds[topic] = []
                self.topics.append(topic)
                
            if self.max['points'] < details['number_of_blocks']:
                self.max['points'] = details['number_of_blocks']
                
            if self.max[topic] < details['number_of_blocks']:
                self.max[topic] = details['number_of_blocks']
            
            self.topicWordIds[topic].append(details['index'])    
            self.points[details['index']] = details
            
        return
    
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
        print('path', self._getPath())
        with open(self._getPath(), 'w') as outfile:
            json.dump(self.data, outfile)
        return
    
    def remove(self):
        file = utility.File(self._getPath())
        file.remove()
        return
 
    def _getPath(self):
    	return self._getFilePath('hierarchy.json', self.path)

