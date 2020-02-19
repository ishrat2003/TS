from .base import Base
import tensorflow_datasets as tfds

class PtToEnTranslate(Base):
    
    def __init__(self, path):
        super().__init__(path)
        self.name = 'pt_to_en_translate'
        return

    def dataset(self):
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = True, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig))
        return examples['train'], examples['validation']
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (pt.numpy() for pt, en in trainingSet)
                
        return (en.numpy() for pt, en in trainingSet)
