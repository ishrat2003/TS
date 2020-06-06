from .meta import Meta
from file.writer import Writer

class Visualization(Writer):
    
    def __init__(self, datasetProcessor, params):
        self.params = params
        self.datasetProcessor = datasetProcessor
        self.path = self.datasetProcessor.getPath()
        
        meta = Meta(self.datasetProcessor)
        self.vocab = meta.loadVocab()
        self.filteredWordsByTopic = {}
        return
    
    def process(self, minNumberOfDocsAppeared = 5):
        if not len(self.vocab):
            return None
        words = self.setFilteredWordsByTopics(minNumberOfDocsAppeared)
        points = self.getPoints()
        self.savePoints()
        return points
    
    def getPoints(self):
        return self.filteredWordsByTopic
    
    def savePoints(self):
        return
    
    def setFilteredWordsByTopics(self, minNumberOfDocsAppeared):
        for word in self.vocab.keys():
            details = self.vocab[word]
            if not self.shouldInclude(details):
                continue
            topic = details['topic']
            occurance = details['count']
            docCount = details['number_of_blocks']
            if docCount < minNumberOfDocsAppeared:
                continue
            if topic not in self.filteredWordsByTopic.keys():
                self.filteredWordsByTopic[topic] = []
            self.filteredWordsByTopic[topic].append(details)
            
        return
    
    def shouldInclude(self, word):
        return True
