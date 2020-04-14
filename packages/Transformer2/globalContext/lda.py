from __future__ import division
import numpy as np
from .vocab import Vocab
from scipy.sparse import csr_matrix
import utility
from sklearn.decomposition import LatentDirichletAllocation
import pickle

class LDA(Vocab):

	def __init__(self, dataset, path):
		super().__init__(dataset, path)
		self.path = path
		self.iteration = 500
		self.verbose = 1
		self.perplexity = 10
		self.numberOfTopics = 10
		self.wordCoOccurenceVector = None
		self.topics = {}
		self.topicFilter = None
		self.vocabSize = 0
		return

	def setNumberOfIterations(self, number):
		self.iteration = number
		return


	def setPerplexity(self, perplexity):
		self.perplexity = perplexity
		return


	def setNumberOfTopics(self, numberOfTopics):
		self.numberOfTopics = numberOfTopics
		return


	def setVerbose(self, verbose):
		self.verbose = verbose
		return


	def setTopicFilter(self, topicNumber):
		self.topicFilter = topicNumber
		return


	def buildWordCoOccurenceVectors(self):
		if not self.dataset:
			print('Failed to prepare word co-occurance matrix. Undefined dataset processor.')
			return

		self._loadVocab()
		self.vocabSize = len(self.vocab)
		print('Vocab size:', self.vocabSize)
		self.wordCoOccurenceVector = np.zeros((self.vocabSize, self.vocabSize))
  
		self._loadSentences()
		print('Total sentences: ', len(self.processedSentences))
		validWordIndexes = self.vocab.keys()
		del self.vocab
		
		if len(self.processedSentences) == 0:
			return

		for sentence in self.processedSentences:
			#print(sentence)
			for word1Index in sentence:
				if word1Index not in validWordIndexes:
					continue
				for word2Index in sentence:
					if (word1Index == word2Index) or (word2Index not in validWordIndexes):
						continue
					else:
						self.wordCoOccurenceVector[word1Index][word2Index] += 1
						print(word1Index, '-', word2Index, ' = ', self.wordCoOccurenceVector[word1Index][word2Index])

		del self.processedSentences
		self.__convertToSparseMatrix()
		self.__saveSparseCsr(self.wordCoOccurenceVector)

		#print(self.wordCoOccurenceVector)
		return self.wordCoOccurenceVector


	def train(self):
		self._load()
		print(self.wordCoOccurenceVector)
		self.wordCoOccurenceVector = np.array(self.wordCoOccurenceVector.toarray())
		print(self.wordCoOccurenceVector.shape)
		lda = LatentDirichletAllocation(n_components=self.numberOfTopics, max_iter=self.iteration, learning_method='online', learning_offset=1.0,random_state=0).fit(self.wordCoOccurenceVector)
		del self.wordCoOccurenceVector
		print(lda.bound_)
  
		self._loadVocab()
		wordScores = {}
		vocabId2Word = {}
		for word in self.vocab:
			vocabId2Word[self.vocab[word]['index']] = word

		self.topics = {}
		for topic_idx, topics in enumerate(lda.components_):
			for i in topics.argsort():
				#print('---------------------')
				if i not in vocabId2Word:
					continue
				word = vocabId2Word[i]
				#print(self.topics.keys())
				#print(word)
				if word in self.topics.keys():
					if self.topics[word] < topics[i]:
						self.topics[word] = topic_idx
				else:
					self.topics[word] = topic_idx

		#print(self.topics)
		self.__saveLdaTopics()
		print("Finished training LDA")
		return

	def _load(self):
		wordCoOccurenceVector = self.__loadSparseCsr()
		if wordCoOccurenceVector is not None:
			self.wordCoOccurenceVector = wordCoOccurenceVector
		self.__loadLdaTopics()
		return


	def __saveSparseCsr(self, vectors):
		filePath = self._getFilePath('word_cooccurence.npz', self.path)
		file = utility.File(filePath)
		file.remove()
		np.savez(filePath, data=vectors.data, indices=vectors.indices, indptr=vectors.indptr, shape=vectors.shape)
		return


	def __saveLdaTopics(self):
		topicsToSave = []
		for word in self.topics:
			topicToSave = {}
			topicToSave['word'] = word
			topicToSave['topic'] = self.topics[word]
			topicsToSave.append(topicToSave)

		self._saveNumpy('lda.npz', topicsToSave)
		return


	def __loadLdaTopics(self):
		topicsFromFiles = self._loadNumpy('lda.npz')
		self.topics = {}

		if topicsFromFiles is not None:
			for fileRef in topicsFromFiles:
				topicsFromFile = topicsFromFiles[fileRef]
				if topicsFromFile is not None:
					for word in topicsFromFile:
						if 'topic' in word.keys():
							self.topics[word['word']] = word['topic']

		return


	def __loadSparseCsr(self):
		filePath = self._getFilePath('word_cooccurence.npz', self.path)
		file = utility.File(filePath)
		if(not file.exists()):
			return None

		loader = np.load(filePath)
		if ((loader['shape'][0] != loader['shape'][1])):
			return None

		return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


	def __convertToSparseMatrix(self):
		vocabSize = self.vocabSize
		data = []
		rows = []
		columns = []
		for i in range(0, vocabSize):
			for j in range(0, vocabSize):
				if self.wordCoOccurenceVector[i][j] > 0:
					rows.append(i)
					columns.append(j)
					data.append(self.wordCoOccurenceVector[i][j])

		self.wordCoOccurenceVector = csr_matrix((data, (rows, columns)), shape=(vocabSize, vocabSize))
		return


	def __saveModel(self, model):
		pickle.dump(model, open(self.__getModelPath(), 'wb'))
		return

	def __getModel(self, model):
		return pickle.load(open(self.__getModelPath(), 'rb'));

	def __getModelPath(self):
		return self._getFilePath('lda_model.sav', self.path)

	def __saveVectorizer(self, model):
		pickle.dump(model, open(self.__getVectorizerPath(), 'wb'))
		return

	def __getVectorizer(self, model):
		return pickle.load(open(self.__getVectorizerPath(), 'rb'));

	def __getVectorizerPath(self):
		return self._getFilePath('lda_vectorizer.sav', self.path)

