from __future__ import absolute_import, division, print_function, unicode_literals
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset
from text.vocab import Vocab
from text.sequences import Sequences


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

testSource = dataProcessor.getGenerator(trainingSet, 'source')
#testTitle = dataProcessor.getGenerator(trainingSet, 'title')
testTarget = dataProcessor.getGenerator(trainingSet, 'target')
print(testSource)
print('==============================================')
# for source in testSource:
#   print('Content: ', source.decode('utf-8'))
  

# for target in testTarget:
#   print('Summary: ', target.decode('utf-8'))
#   print('----------')


for (batch, (source, target)) in enumerate(trainingSet):
  print('Batch: ', batch)

  print('Content:::::::::::: ', source.numpy().decode('utf-8'))
  print('Summary:::::::::::: ', target.numpy().decode('utf-8'))

