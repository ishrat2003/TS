from .meta import Meta
from .visualization import Visualization
import numpy

class CWR(Visualization):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        self.totalAngle = 360
        self.anglePerTopic = 360
        self.startAngle = 0
        self.topicFilter = None
        self.maxRadius = 0
        return
    
    def setTopicFilter(self, topicName):
        self.topicFilter = topicName
        return
    
    def setTotalAngle(self, angle):
        self.totalAngle = angle
        return

    def setStartAngle(self, start):
        self.startAngle = start
        return

    def getPoints(self):
        topics = self.filteredWordsByTopic.keys()
        totalTopics = len(topics)
        
        if not self.topicFilter:
            self.anglePerTopic = self.totalAngle / totalTopics
        
        self.startAngle = 0
        self.setMaximumRadius()
        
        self.points = []
        
        for topic in topics:
            if self.topicFilter and (topic != self.topicFilter):
                continue
            self.appendPoints(self.filteredWordsByTopic[topic], topic)
            self.startAngle += self.anglePerTopic
            
        return self.points
    
    def appendPoints(self, words, topic):
        self.pointsFile = self.getFile('cwr_gc.csv')
        
        totalWords = len(words)
        thetaIncrement = self.anglePerTopic / totalWords
        currentTheta = self.startAngle
        
        for word in words:
            point = word
            point['radius'] =  self.maxRadius - word['number_of_blocks']
            point['topic'] = topic
            point['theta'] = currentTheta
            currentTheta += thetaIncrement
            point['x'] = word['radius'] * numpy.cos(numpy.deg2rad(point['theta']))
            point['y'] = word['radius'] * numpy.sin(numpy.deg2rad(point['theta']))
            self.points.append(point)
            self.pointsFile.write(point)

        return points
    
    def setMaximumRadius(self):
        for word in self.vocab.keys():
            if self.maxRadius < self.vocab[word]['number_of_blocks']:
                self.maxRadius = self.vocab[word]['number_of_blocks'] + 10
        return
    
    def savePoints(self):
        print(self.points)
        self._saveInPickel(self._getPointsPath(), self.points)
        return
    
    def loadPoints(self):
        self.points = self._getPointsPath(self._getDocLcsPath())
        return self.points

    def _getPointsPath(self):
    	return self._getFilePath('cwr_gc_pk.sav', self.path)
