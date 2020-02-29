from .base import Base
import tensorflow_datasets as tfds

class Newsroom(Base):
    
    def __init__(self, path):
        super().__init__(path, 'newsroom')
        return

    def dataset(self):
        return self._load('newsroom')
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (text.numpy() for text, summary in trainingSet)
        
        return (summary.numpy() for text, summary in trainingSet)
    
