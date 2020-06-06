import sys
packagesPath = "/content/drive/My Drive/Colab Notebooks/packages/TextMining"
sys.path.append(packagesPath)

import globalContext as GC
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


logging.basicConfig(level=logging.INFO)

logging.info("# Building global context")
logging.info("# 1. Loading script params ")
logging.info("# ================================")
scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

logging.info("# 2. Preprocessing data")
logging.info("# ================================")

dataset = Dataset(params.dataset_name, params.data_directory)
datasetToProcess = dataset.get(float(params.dataset_percentage), int(params.total_items))

if not datasetToProcess:
    logging.error('No dataset found')
    sys.exit()

vocabPath = os.path.join(datasetToProcess.path, params.dataset_name)
datasetPipeline = datasetToProcess.getTrainingSet().take(1) 

print('datasetPipeline:::', datasetPipeline)
print('path', vocabPath)
if os.path.exists(vocabPath + '.subwords'):
    logging.info("# Loading vocab")
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocabPath)
else:
    logging.info("# Creating vocab")
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(datasetToProcess.getGenerator(datasetPipeline), 10000)
    tokenizer.save_to_file(vocabPath)

tokenizedDataSet = datasetToProcess.getTokenizedDataset(datasetPipeline, tokenizer)
print('tokenizedDataSet', tokenizedDataSet)
embedding_dim=2
vocab_size = 2
model = keras.Sequential([
  layers.Embedding(3, 2),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()


# data = dataset.take(1)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_batches = tokenizedDataSet.shuffle(1000).padded_batch(10)
print(tokenizedDataSet);

train_batches = [[1, 2], [1, 3]]
train_batches = tf.convert_to_tensor(train_batches, dtype=tf.float32) 

validation_batches = [[1, 2], [1, 3]]
validation_batches = tf.convert_to_tensor(train_batches, dtype=tf.float32)

history = model.fit(
    train_batches,
    epochs=1,
    validation_data= validation_batches
, validation_steps=20)


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


for item in datasetToProcess.getGenerator(datasetPipeline.take(1)):
    print(item)


print('Finished')




