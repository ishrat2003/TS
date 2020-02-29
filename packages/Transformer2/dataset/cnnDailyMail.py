from .base import Base
import tensorflow_datasets as tfds

class CNNDailyMail(Base):
    
    def __init__(self, path):
        super().__init__(path, 'cnn_dailymail/plain_text')
        return

    def dataset(self):
        return self._load('cnn_dailymail/plain_text')
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (article.numpy() for article, highlight in trainingSet)
        
        return (highlight.numpy() for article, highlight in trainingSet)
    
