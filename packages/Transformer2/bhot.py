from __future__ import absolute_import, division, print_function, unicode_literals
import os, logging, sys
from lc.evaluate import Evaluate
from utility.file import File
from params.core import Core as Params

logging.basicConfig(level=logging.INFO)
scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

book = File('/content/drive/My Drive/Colab Notebooks/data/bhot/bhotall')
sourceText = book.read()
#http://www.supersummary.com/a-brief-history-of-time/summary/
bookSummary = File('/content/drive/My Drive/Colab Notebooks/data/bhot/bhotSummary2')
targetText = bookSummary.read()

evaluationProcessor = Evaluate(None, params)
allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
#allowedTypes = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
#allowedTypes = ['JJ', 'JJR', 'JJS']
#allowedTypes = ['RB', 'RBR', 'RBS']
#allowedTypes = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
#allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
evaluationProcessor.setAllowedTypes(allowedTypes)
evaluationProcessor.initInfo()
evaluationProcessor.processItem(0, sourceText, targetText)
evaluationProcessor.summarizeInfo()


