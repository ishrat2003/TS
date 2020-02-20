import os
import tensorflow_datasets as tfds

class Base:
    
    def __init__(self, path, name = None):
        self.directoryPath = path
        self.name = name
        if name:
            self.path = os.path.join(path, self.name)
        else:
            self.path = path
        return
    
    def get(self):
        self.dlConfig = tfds.download.DownloadConfig(manual_dir=self.directoryPath)
        return self.dataset()
    
    def getProcessedPath(self):
        return self.path
    
    def getVocabPath(self):
        return os.path.join(self.path, "vocab")

    def dataset(self):
        return None, None
    
    def getGenerator(self, trainingSet, type = 'source'):
        return None
    