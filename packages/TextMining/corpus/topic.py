from .meta import Meta
from .visualization import Visualization
import numpy
import utility
import json
import math

class Topic(Visualization):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        self.totalZones = 10
        self.topics = []
        self.count = {}
        self.setMaximumRadius()
        self.setGap()
        self.topicFile = self.getFile('stream-topic.csv')
        return
    
    def setGap(self):
        self.gap = self.maxRadius / self.totalZones
        return
    
    def setTotalZones(self, totalZones):
        self.totalZones = totalZones
        return
    
    def setMaximumRadius(self):
        self.maxRadius = 0
        for word in self.vocab.keys():
            if self.maxRadius < self.vocab[word]['number_of_blocks']:
                self.maxRadius = self.vocab[word]['number_of_blocks'] + 10
        return

    def process(self, minNumberOfDocsAppeared = 5):
        if not len(self.vocab):
            return

        self.preparePoints()
        for zone in self.count.keys():
            row = {
                'zone': zone
            }
            for topic in self.topics:
                row[topic] = 0
                
            for topic in self.count[zone].keys():
                row[topic] = self.count[zone][topic]

            self.topicFile.write(row)
        return
    
    def preparePoints(self):
        for word in self.vocab.keys():
            details = self.vocab[word]
            zone = math.ceil((self.maxRadius - details['number_of_blocks']) / self.gap)
            topic = details['topic']
            if topic not in self.topics:
                self.topics.append(topic)
            if zone not in self.count.keys():
                self.count[zone] = {}
            if topic not in self.count[zone].keys():
                self.count[zone][topic] = 0
            self.count[zone][topic] += 1
        return
    
    
