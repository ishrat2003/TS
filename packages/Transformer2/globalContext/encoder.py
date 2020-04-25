import tensorflow_datasets as tfds

class Encoder(tfds.features.text.SubwordTextEncoder):
    
    def encode(self, s):
        print('Encoder is called:::: ', s)
        return super().encode(s)
