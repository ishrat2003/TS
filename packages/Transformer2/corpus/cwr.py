from .meta import Meta
from .visualization import Visualization
import numpy
import utility
import math

class CWR(Visualization):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        self.totalAngle = 360
        self.anglePerTopic = 360
        self.startAngle = 0
        self.topicFilter = None
        self.maxRadius = 0
        self.totalZones = 10
        self.totalFonts = 100
        self.zoneGap = 0
        self.fontGap = 0
        return
    
    def setTotalZones(self, totalZones):
        self.totalZones = totalZones
        return
    
    def setTotalFonts(self, totalZones):
        self.totalFonts = totalZones
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
    
    def setMaximumRadius(self):
        for word in self.vocab.keys():
            if self.maxRadius < self.vocab[word]['number_of_blocks']:
                self.maxRadius = self.vocab[word]['number_of_blocks'] + 10
        return
    
    def setGaps(self):
        self.zoneGap = self.maxRadius / self.totalZones
        self.fontGap = self.maxRadius / self.totalFonts
        return

    def getPoints(self):
        topics = self.filteredWordsByTopic.keys()
        totalTopics = len(topics)
        
        if not self.topicFilter:
            self.anglePerTopic = self.totalAngle / totalTopics
        
        self.startAngle = 0
        self.setMaximumRadius()
        self.setGaps()
        
        self.points = []
        self.pointsFile = self.getFile('cwr_gc.csv')
        
        for topic in topics:
            if self.topicFilter and (topic != self.topicFilter):
                continue
            self.appendPoints(self.filteredWordsByTopic[topic], topic)
            self.startAngle += self.anglePerTopic
            
        return self.points
    
    def appendPoints(self, words, topic):
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
            point['zone'] = math.floor(point['radius'] / self.zoneGap)
            point['font_size'] = math.floor(point['radius'] / self.fontGap)
            self.points.append(point)
            self.pointsFile.write(point)

        return
    
    def savePoints(self):
        print(self.points)
        self._saveInPickel(self._getPointsPath(), self.points)
        return
    
    def loadPoints(self):
        self.points = self._getPointsPath(self._getPointsPath())
        return self.points
    
    def generateTopicFiles(self):
        topics = list(self.filteredWordsByTopic.keys())
        self.removeCategorizedFiles(topics)
        topicFiles = self.getCategorizedFiles(topics)
        for topic in topics:
            points = self.filteredWordsByTopic[topic]
            for point in points:
               topicFiles[topic].write(point) 
        return
    
    def generateSentimentFiles (self):
        sentiments = ["normal", "positive", "negative"]
        self.removeCategorizedFiles(sentiments)
        
        sentimentFiles = self.getCategorizedFiles(sentiments)
        for point in self.points:
            fileKey = point['sentiment']
            sentimentFiles[fileKey].write(point)
        return
    
    def getCategorizedFiles(self, keys):
        files = {}
        for key in keys:
            files[key] = self.getFile( key + '_cwr_gc.csv')
        return files
    
    def removeCategorizedFiles(self, keys):
        files = self.getCategorizedFiles(keys)
        for key in keys:
            files[key].remove()
        return
    
    def remove(self):
        file = utility.File(self._getPointsPath())
        file.remove()
        csvFile = self.getFile('cwr_gc.csv')
        file.remove()
        return

    def _getPointsPath(self):
    	return self._getFilePath('cwr_gc_pk.sav', self.path)

