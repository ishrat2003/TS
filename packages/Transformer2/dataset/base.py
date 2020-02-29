import os
import tensorflow_datasets as tfds

class Base:
    
    def __init__(self, path, name = None):
        self.directoryPath = path
        self.name = name
        self.splitPercentage = 100
        self.metadata = None
        if name:
            self.path = os.path.join(path, self.name)
        else:
            self.path = path
        return
    
    def setSplitPercentage(self, percentage = 100):
        if ((percentage < 0) or (percentage > 100)):
            percentage = 100
        self.splitPercentage = percentage
        return
    
    def get(self):
        self.dlConfig = tfds.download.DownloadConfig(manual_dir=self.directoryPath)
        return self.dataset()
    
    def getMetadata(self):
        return self.metadata
    
    def getProcessedPath(self):
        return self.path
    
    def getVocabPath(self):
        return os.path.join(self.path, "vocab")

    def dataset(self):
        return None, None
    
    def getGenerator(self, trainingSet, type = 'source'):
        return None
    
    def _load(self, key): 
        readInstructions = [
            tfds.core.ReadInstruction('train', to = self.splitPercentage, unit='%'),
            tfds.core.ReadInstruction('validation', to = self.splitPercentage, unit='%'),
        ]      
        data, self.metadata = tfds.load(key, 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = True, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig),
            split = readInstructions)
        
        return data[0], data[1]
    