from __future__ import absolute_import, division, print_function, unicode_literals
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset
from lc.summaryEvaluate import SummaryEvaluate
from lc.titleEvaluate import TitleEvaluate

logging.basicConfig(level=logging.INFO)

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

if params.type == 'title':
    evaluationProcessor = TitleEvaluate(datasetToProcess, params)
else:
    evaluationProcessor = SummaryEvaluate(datasetToProcess, params)
    
logging.info("# 3. Evaluating")
logging.info("# ================================")

evaluationProcessor.process()

logging.info("# 4. Finished")

