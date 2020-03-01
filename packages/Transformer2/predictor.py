from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os, logging, sys
import time
import numpy as np
import matplotlib.pyplot as plt

from params.core import Core as Params
from dataset.core import Core as Dataset
from text.vocab import Vocab

from predictor.sequence import Sequence as TransformerPredictor


logging.basicConfig(level=logging.INFO)

logging.info("# 1. Loading script params ")
logging.info("# ================================")
scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

logging.info("# 2. Preprocessing sequences")
logging.info("# ================================")
logging.info("# 2.1. Loading raw dataset")
dataset = Dataset(params.dataset_name, params.data_directory)
dataProcessor = dataset.get()

trainingSet, validationSet = dataProcessor.get()
if not trainingSet or not validationSet:
    logging.error('No dataset found')
    sys.exit()

logging.info("# 2.2. Preparing vocab (source + target)")
vocabProcessor = Vocab(dataProcessor)
logging.info("# Preparing vocab (source)")
tokenizerSource = vocabProcessor.get(trainingSet, 'source')
logging.info("# Preparing vocab (target)")
tokenizerTarget = vocabProcessor.get(trainingSet, 'target')

logging.info("# 3. Predict")
logging.info("# ================================")

input = "este é um problema que temos que resolver.";
predictor = TransformerPredictor(params, tokenizerSource, tokenizerTarget)
output = predictor.process(input)

print('Input: ', input)
print('Output: ', output)

logging.info('Finished')
logging.info("# ================================")

