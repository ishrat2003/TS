from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .store import Store
from operator import itemgetter 
import pickle
from utility.file import File
import os

class LDA(Store):

	def __init__(self, datasetProcessor, path):
		self.datasetProcessor = datasetProcessor
		self.path = path
		self.iteration = 1000
		self.verbose = 1
		self.perplexity = 10
		self.numberOfTopics = 10
		self.totalTopFrequencyWords = 1000
		self.combinedTopics = None
		self.model = self.__getModel()
		self.vectorizer = self.__getVectorizer()
		self.processedDocumentsTopics = self.__getDocumentTopics()
		self.allTopics = self.getTopics()
		self.documentLabels = [] 
		return

	def setNumberOfIterations(self, number):
		self.iteration = number
		return

	def setNumberOfTotalTopFrequencyWord(self, vocabSize):
		self.totalTopFrequencyWords = vocabSize
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

	def getCountVectorizer(self):
		dataset = self.datasetProcessor.get()
		if not dataset:
			print('Failed to prepare word co-occurance matrix. Undefined dataset processor.')
			return

		documents = []
		for item in dataset:
			itemData, label = item
			self.documentLabels.append(label.numpy().decode("utf-8"))
			bagOfWords = self.datasetProcessor.getText(itemData.numpy())
			documents.append(bagOfWords)
			# print('---------------------')
			# print(bagOfWords)

		# max_dff [0.0, 1.0] = When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). 
		# min_df [0.0, 1.0] = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
		self.vectorizer = CountVectorizer(max_df = 1.0, min_df = 2, max_features = self.totalTopFrequencyWords, stop_words='english')
		tf = self.vectorizer.fit_transform(documents)
		self.features = self.vectorizer.get_feature_names()

		self.__saveVectorizer(self.vectorizer)
		del documents # Freeing memory
		return tf

	def train(self):
		tf = self.getCountVectorizer()
		self.model = LatentDirichletAllocation(n_components=self.numberOfTopics, max_iter=self.iteration, learning_method='online', learning_offset=1.0,random_state=0).fit(tf)

		self.__saveModel(self.model)
		words = self.saveWords(self.model)
		self.saveTopics(words)

		documentTopics = self.model.transform(tf)
		self.__saveDocumentTopics(documentTopics)
		return

	def getTopics(self):
		return self.getFileContent('topics-lda.npz')

	def getWords(self):
		return self.getFileContent('words-lda.npz')
    
	def getFileContent(self, filename):
		fromFiles = self._loadNumpy(filename, self.path)

		if fromFiles is not None:
			for fileRef in fromFiles:
				dataFromFile = fromFiles[fileRef]
				if dataFromFile is not None:
					return dataFromFile

		return None

	def saveTopics(self, words):
		topics = {}
		for stemmedWord in words.keys():
			dominantTopicIndex = self.getDominantTopicIndex(words[stemmedWord])
			if dominantTopicIndex not in topics.keys():
				topics[dominantTopicIndex] = []
			if dominantTopicIndex:
				topics[dominantTopicIndex].append(stemmedWord)

		self.save(topics, 'topics-lda.npz')
		self.topics = topics
		return

	def getDominantTopicIndex(self, word):
		dominantTopicIndex = None	
		topicScore = 0
		for topicIndex in word.keys():
			if not dominantTopicIndex:
				dominantTopicIndex = topicIndex
			elif topicScore < word[topicIndex]:
				dominantTopicIndex = topicIndex

			topicScore = word[dominantTopicIndex]

		return dominantTopicIndex

	def saveWords(self, model):
		words = {}
		
		for topicIndex, topics in enumerate(model.components_):
			# numpy argsort returns the indices that would sort an array.
			for wordIndex in topics.argsort():
				stemmedWord = self.features[wordIndex]
				if stemmedWord not in words.keys():
					words[stemmedWord] = {}

				words[stemmedWord][topicIndex] = topics[wordIndex] # topicIndex => Score

		self.save(words, 'words-lda.npz')
		return words

	def save(self, items, fileName):
		self._saveNumpy(fileName, items, self.path)
		return

	def printTopics(self):
		print(self.getTopics())
		return

	def predictedTopics(self, words, label, limit = 5, limitPerTopic = 500):
		print('combined topics    ', self.getCombinedTopics(limitPerTopic))
		itemTopics = self.intersection(self.getCombinedTopics(limitPerTopic), words)
		print('itemTopics', itemTopics)
		#itemTopics = self.intersection(self.getDominantTopicTerms(label), words)
		if not len(itemTopics):
			return []
		
		wordWeights = self.getWordsMaxWeightForTheDominantTopic(itemTopics)
		print('wordWeights', wordWeights)
		sortedWordWeights = sorted(wordWeights.items(), key = itemgetter(1), reverse = True)
		print('sortedWordWeights', sortedWordWeights)
		return [key for (key, value) in sortedWordWeights[:limit]]

	def getWordsMaxWeightForTheDominantTopic(self, itemTopics):
		wordWeights = {}
		words = self.getWords().tolist()

		for word in itemTopics:
			if word not in words.keys():
				continue
			wordWeights[word] = 0
			for topicIndex in words[word].keys():
				if words[word][topicIndex] > wordWeights[word]:
					wordWeights[word] = words[word][topicIndex]
  
		del words
		return wordWeights

	def getDominantTopicTerms(self, label):
		topicIndex = self.processedDocumentsTopics[label]['dominant_topic_indexes'][0]
		print(topicIndex)
		allTopics = self.allTopics.tolist()
		print(allTopics)
		return allTopics[topicIndex]

	def getCombinedTopics(self, limitPerTopic):
		if self.combinedTopics:
			return self.combinedTopics
	
		allTopics = self.allTopics.tolist()

		self.combinedTopics = []
		for topicIndex in allTopics.keys():
			topTopics = allTopics[topicIndex][:limitPerTopic]
			self.combinedTopics += topTopics

		del allTopics
		return self.combinedTopics

	def intersection(self, list1, list2):
		return list(set(list1) & set(list2))

	def __saveDocumentTopics(self, documentsTopics):
		processedDocumentsTopics = {}
		index = 0
		for documentTopics in documentsTopics:
			processedDocumentsTopics[self.documentLabels[index]] = {}
			processedDocumentsTopics[self.documentLabels[index]]['topic_score'] = documentTopics
			processedDocumentsTopics[self.documentLabels[index]]['dominant_topic_indexes'] = documentTopics.argsort()
			index += 1
			
		self.__saveInPickel(self.__getDocumentTopicslPath(), processedDocumentsTopics)
		return

	def __getDocumentTopics(self):
		return self.__getFromPickel(self.__getDocumentTopicslPath())

	def __getDocumentTopicslPath(self):
		return self._getFilePath('lda_document_topics.sav', self.path)

	def __saveModel(self, model):
		self.__saveInPickel(self.__getModelPath(), model)
		return

	def __getModel(self):
		return self.__getFromPickel(self.__getModelPath())

	def __getModelPath(self):
		return self._getFilePath('lda_model.sav', self.path)

	def __saveVectorizer(self, vectorizer):
		pickle.dump(vectorizer, open(self.__getVectorizerPath(), 'wb'))
		return

	def __getVectorizer(self):
		path = self.__getVectorizerPath()
		return self.__getFromPickel(path)

	def __getVectorizerPath(self):
		return self._getFilePath('lda_vectorizer.sav', self.path)

	def __getFromPickel(self, filePath):
		file = File(filePath)
		if file.exists():
			return pickle.load(open(filePath, 'rb'));
		return None

	def __saveInPickel(self, filePath, model):
		file = File(filePath)
		if file.exists():
			file.remove()
		pickle.dump(model, open(filePath, 'wb'))
		return
