import os, sys
import nltk
import string
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import defaultdict


class InfoRetrieval(object):

	def __init__(self, questionDoc, document):
		self.questionDoc = questionDoc
		self.document = document
		self.questionList = []
		self.wordWeights = dict()
		#self.wnl = WordNetLemmatizer()
		self.portStem = PorterStemmer()

	def getQuestions(self):
		with open(self.questionDoc, "rt") as fileContents:
			content = fileContents.read()
			fileContents.close()
		for question in content.splitlines():
			self.questionList.append(question)


	def getSentences(self):
		with open(self.document, "rt") as fileContents:
			content = fileContents.read()
			fileContents.close()

		possibleSentences = nltk.tokenize.sent_tokenize(content.replace("\n", " . "))
		return possibleSentences

	# Take a sentence and return its stripped down, stemmed form
	def lemmatize(self, s):
		# remove punctuation in python 3.0+
		translator = str.maketrans('', '', string.punctuation)
		s2 = s.translate(translator)
		# split sentence into words via tokenization
		s3 = nltk.tokenize.word_tokenize(s2)
		lemString = ""
		# actually lemmatize
		for word in s3:
			lemString += self.portStem.stem(word.lower())
			lemString += " "
		return lemString

	def weightWords(self):
		weightDict = dict()
		with open(self.document, "rt") as fullDoc:
			f = fullDoc.read()
			fullDoc.close()
		# want to lemmatize to equalize word polymorphisms
		allWords = self.lemmatize(f).split()
		for word in allWords:
			if (word in weightDict):
				weightDict[word] = weightDict[word] + 1
			else:
				weightDict[word] = 1
		return weightDict

	def getTargetSentence(self, question, sentenceList):
		possibleTargetSentences = []
		maxScore = 0
		bestSentence = ""
		lemQuestion = self.lemmatize(question)
		qWords = lemQuestion.split()
		for sentence in sentenceList:
			lemSentence = self.lemmatize(sentence)
			questionRank = 0
			for word in lemSentence.split():
				# if a word in the sentence matches one in the question, count it
				if (word in qWords):
					questionRank += 1.0/self.wordWeights[word]
			if (questionRank > maxScore):
				bestSentence = sentence
				maxScore = questionRank
		return bestSentence

	def run(self):
		self.wordWeights = self.weightWords() # find word counts for inverse frequency calculations
		self.getQuestions() # adds to questionlist
		sentenceList = self.getSentences()
		with open("ChosenSentences.txt", "w") as chosen:
			chosen.write("")
		for question in self.questionList:
			target = self.getTargetSentence(question, sentenceList)
			with open("ChosenSentences.txt", "a") as chosen:
				chosen.write(target)
				chosen.write("\n")

if __name__ == '__main__':
	InfoRetrieval(questionDoc=sys.argv[1], document=sys.argv[2]).run()



