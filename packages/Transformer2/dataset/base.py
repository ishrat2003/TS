import os
import tensorflow_datasets as tfds

class Base:
    
    def __init__(self, path):
        self.name = 'default'
        self.path = path
        return
    
    def get(self):
        self.dlConfig = tfds.download.DownloadConfig(manual_dir=self.path)
        return self.dataset()
    
    def getProcessedPath(self):
        return os.path.join(self.path, "processed", self.name)
    
    def getVocabPath(self):
        return os.path.join(self.path, "processed", self.name, "vocab")

    def dataset(self):
        return None, None
    
    def getGenerator(self, trainingSet, type = 'source'):
        return None
    