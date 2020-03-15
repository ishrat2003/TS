from __future__ import absolute_import, division, print_function, unicode_literals
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset
from lc.evaluate import Evaluate

logging.basicConfig(level=logging.INFO)

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

evaluationProcessor = Evaluate(trainingSet, params)
allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
#allowedTypes = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
#allowedTypes = ['JJ', 'JJR', 'JJS']
#allowedTypes = ['RB', 'RBR', 'RBS']
#allowedTypes = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
#allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
evaluationProcessor.setAllowedTypes(allowedTypes)
evaluationProcessor.process()


