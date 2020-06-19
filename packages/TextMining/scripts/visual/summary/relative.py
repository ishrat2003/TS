import sys
packagesPath = "/content/drive/My Drive/Colab Notebooks/packages/TextMining"
sys.path.append(packagesPath)

import os, logging
from params.core import Core as Params
from dataset.core import Core as Dataset
from visual_summaries.relative import Relative

logging.basicConfig(level=logging.INFO)
logging.info("# This script generates meta information about the corpus and summaries")

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

manager = Relative(datasetToProcess)
manager.remove()
manager.process()

print('Finished')


