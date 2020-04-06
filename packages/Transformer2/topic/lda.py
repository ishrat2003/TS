from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .store import Store
from operator import itemgetter 

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
		self.model = None
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
			bagOfWords = self.datasetProcessor.getLCTWordsOccurredMoreThanMinCount(item)
			documents.append(bagOfWords)

		# max_dff [0.0, 1.0] = When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). 
		# min_df [0.0, 1.0] = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
		tfVectorizer = CountVectorizer(max_df = 0.95, min_df = 2, max_features = self.totalTopFrequencyWords, stop_words='english')
		tf = tfVectorizer.fit_transform(documents)
		self.features = tfVectorizer.get_feature_names()

		del documents # Freeing memory
		return tf

	def train(self):
		tf = self.getCountVectorizer()
		self.model = LatentDirichletAllocation(n_components=self.numberOfTopics, max_iter=self.iteration, learning_method='online', learning_offset=1.0,random_state=0).fit(tf)

		words = self.saveWords(self.model)
		self.saveTopics(words)
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

	def predictedTopics(self, words, limit = 5, limitPerTopic = 100):
		itemTopics = self.intersection(self.getCombinedTopics(limitPerTopic), words)
		if not len(itemTopics):
			return []
		
		wordWeights = self.getWordsMaxWeightForTheDominantTopic(itemTopics)
		sortedWordWeights = sorted(wordWeights.items(), key = itemgetter(1), reverse = True)
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

	def getCombinedTopics(self, limitPerTopic):
		if self.combinedTopics:
			return self.combinedTopics

		allTopics = self.getTopics()
		allTopics = allTopics.tolist()

		self.combinedTopics = []
		for topicIndex in allTopics.keys():
			topTopics = allTopics[topicIndex][:limitPerTopic]
			self.combinedTopics += topTopics

		del allTopics
		return self.combinedTopics

	def intersection(self, list1, list2):
		return list(set(list1) & set(list2)) 
