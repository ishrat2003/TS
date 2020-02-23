from .base import Base
import tensorflow_datasets as tfds

class ScientificPapers(Base):
    
    def __init__(self, path, config):
        self.config = config
        super().__init__(path, config)
        return

    def dataset(self):
        data, metadata = tfds.load(self.config, 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = True, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig))
        
        return data['train'], data['validation']
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (article.numpy() for article, abstract in trainingSet)
        
        return (abstract.numpy() for article, abstract in trainingSet)
    
