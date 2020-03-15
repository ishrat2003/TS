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
from text.sequences import Sequences

from layers.transformer import Transformer
from predictor.sequence import Sequence as TransformerPredictor
from network.trainer import Trainer
from evaluation.rouge import Rouge

logging.basicConfig(level=logging.INFO)
# tf.executing_eagerly()

logging.info("# 1. Loading script params ")
logging.info("# ================================")
scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

logging.info("# 2. Preprocessing sequences")
logging.info("# ================================")
logging.info("# 2.1. Loading raw dataset (" + str(params.dataset_percentage) + "%)")
dataset = Dataset(params.dataset_name, params.data_directory)
dataProcessor = dataset.get(float(params.dataset_percentage), int(params.total_items))

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
logging.info("# Sample output using source vocab")
sampleString = 'This news is great'
vocabProcessor.printSample(tokenizerSource, sampleString)
logging.info("# Sample output using target vocab")
vocabProcessor.printSample(tokenizerTarget, sampleString)

logging.info("# 2.3. Preparing sequences (source (text) + target (summary))")
sequencesProcesser = Sequences(params, tokenizerSource, tokenizerTarget)
trainingSequences = sequencesProcesser.getTokenizedDataset(trainingSet)
#validaitionSequences = sequencesProcesser.getTokenizedDataset(validationSet, True)

# logging.info("# Sample sequences for training dataset")
# sequencesProcesser.printSample(trainingSequences)
# logging.info("# Sample sequences for validation dataset")
# sequencesProcesser.printSample(validaitionSequences)

logging.info("# 3. Training")
logging.info("# ================================")
logging.info("# 3.1. Setting layers")

trainer = Trainer(params)
inputVocabSize = tokenizerSource.vocab_size + 2
targetVocabSize = tokenizerTarget.vocab_size + 2
transformer = Transformer(params, inputVocabSize, targetVocabSize, 
                          pe_input=inputVocabSize, 
                          pe_target=targetVocabSize)

trainer.setModel(transformer)
trainer.setCheckpoint(params.checkpoint_directory + '/' + params.dataset_name)
trainer.setTensorboard(params.log_directory + '/' + params.dataset_name)

logging.info("# 3.2. Setting validation set")
trainer.setValidationDataset(validationSet)

logging.info("# 3.2. Setting predictor")
predictor = TransformerPredictor(params, tokenizerSource, tokenizerTarget)
trainer.setPredictor(predictor)

logging.info("# 3.2. Setting predictor")
evaluator = Rouge(params)
trainer.setEvaluator(evaluator)

logging.info("# 3.3. Training & Evaluation")
trainer.process(int(params.epochs), trainingSequences)
  
logging.info('Finished')
logging.info("# ================================")

