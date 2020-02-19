from .base import Base
import tensorflow_datasets as tfds

class Newsroom(Base):
    
    def __init__(self, path):
        super().__init__(path)
        self.name = 'newsroom'
        return

    def dataset(self):
        data, metadata = tfds.load('newsroom', 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = True, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig))
        
        return data['train'], data['validation']
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (text.numpy() for text, summary in trainingSet)
        
        return (summary.numpy() for text, summary in trainingSet)
    
