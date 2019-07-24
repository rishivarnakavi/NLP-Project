#!/usr/bin/python

# Philip Dominici, Ryan Sickles

# April 2017
# Question answering program for 11441 semester project


import os, sys
import re
import string
import math
import numpy
import nltk
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import timex

class AnsweringMachine(object):

	def __init__(self, questionDoc, sentenceDoc):
		# unpack question(s) and sentence(s) from txt files to strings
		self.questionList = []
		with open(questionDoc, 'r') as f:
			content = f.read()
			f.close()
		for question in content.strip().splitlines():
			self.questionList.append(question)
		self.sentenceList = []
		with open(sentenceDoc, 'r') as f:
			content = f.read()
			f.close()
		for question in content.strip().splitlines():
			self.sentenceList.append(question)
		# etc.
		self.wh = "who what when where which"
		self.portStem = PorterStemmer()

	# remove punctuation
	# input: STRING
	# output: STRING which is stripped
	def clean(self, s):	
		preppedString = s.strip()
		preppedString = s.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").replace(";", " ;")
		preppedString = timex.timexTag(preppedString)
		return(preppedString)

	# pass in a string to be tagged in NER
	# input: a STRING, the sentence to be evaluated
	# return: a LIST of TUPLES, each containing the word and its named entity
	def ner(self, sentence):
		rawEntities = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence))).pos() # tree object
		entities = list()
		for ent in rawEntities:
			pureTuple = (ent[0][0], ent[1])
			entities.append(pureTuple)
		return(entities)


	def answerQuestion(self, question, sentence):
		# all the question tags we will cownsider
		searchObj = re.findall(r'did|was|is|who|what|where|when|how|which|why', question, re.I)
		qType = None
		# if only one question-word, take that
		if (len(searchObj) == 1):
			qType = searchObj[0]
		# if >1 question-word, take first wh-words if exists, otherwise just take the first word
		else:
			for word in searchObj:
				if ((word in self.wh) and (qType == None)):
					qType = word
			if (qType == None): qType = searchObj[0]
		# note that we are working in lower case to identify question types
		if (qType.lower() in self.wh):
			answer = self.answerWh(qType.lower(), question, sentence)
		elif (qType.lower() == "why"):
			answer = self.answerWhy(question, sentence)
		else:
			answer = self.answerBinary(question, sentence)
		print(answer)

	# consider binary (yes or no) questions
	def answerBinary(self,question,sentence):
		#first tag all
		answer = "Yes"
		question_tags = nltk.word_tokenize(question)
		q_tags = nltk.pos_tag(question_tags)
		q_identified_words = []
		for word,tag in q_tags:
			if("NN" in tag or "J" in tag):
				q_identified_words.append(word)
		target_sentence_tags = nltk.word_tokenize(sentence)
		s_tags = nltk.pos_tag(target_sentence_tags)
		# print(s_tags)
		negative_words = ["does not", "is not", "not", "don't", "isn't", "is not"]
		is_negative = False
		for word,tag in s_tags:
			if(word in q_identified_words):
				answer = "Yes"
			if(word in negative_words):
				is_negative = True
		if(is_negative):
			answer = "No"
		return(answer)

	# answer why by trying to pick out substring
	def answerWhy(self, question, sentence):
		answer = ""
		# verbs that don't need consideration
		detVbs = ["does", "did", "do", "is", "was", "will", "were", "are"]
		vbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
		nns = ["NN", "NNS", "NNPS", "NNP", "PRP"]
		ajs = ["JJ", "JJS", "JJR", "IN"]
		qPOS = nltk.pos_tag(nltk.word_tokenize(question))
		sentPOS = nltk.pos_tag(nltk.word_tokenize(sentence))
		coreQ = []
		for i in qPOS:
			if ((i[0] not in detVbs) and (i[1] in vbs+nns+ajs)):
				coreQ.append(self.portStem.stem(i[0]))
		coreLocs = [] # in-question verbs/nouns
		otherLocs = [] # non-question verbs/nouns
		for j in range(0, len(sentPOS)):
			curWord = self.portStem.stem(sentPOS[j][0])
			curTag = sentPOS[j][1]
			if (curWord in coreQ):
				coreLocs.append(j)
			if ((curWord not in coreQ+detVbs) and (curTag in vbs+nns+ajs)):
				otherLocs.append(j)
		if (len(otherLocs) == 0):
			return(sentence)
		# prefer to return answers after the subject 
		startPhrase = ""
		if (max(coreLocs) < max(otherLocs)):
			ansRange = [i for i in otherLocs if i > max(coreLocs)]
			ansRange = range(min(ansRange), max(ansRange)+1)
		# if only have content before, return that
		if (min(coreLocs) > min(otherLocs)):
			ansRange = [i for i in otherLocs if i < min(coreLocs)]
			ansRange = range(min(ansRange), max(ansRange)+1)
			if (sentPOS[(max(ansRange)-1 in nns)][0] and (sentPOS[max(ansRange)][0] in vbs)):
				ansRange = range(min(ansRange), max(ansRange)-1)				
		# find out if we need to format extra
		ansWords = [sentPOS[i][0] for i in ansRange]
		if (sentPOS[min(ansRange)][1] in nns):
			startPhrase = "Because of "
		if (sentPOS[min(ansRange)][1] in vbs):
			startPhrase = "To "
		answer = startPhrase + " ".join(ansWords)
		return(answer)

	# consider wh- (subject specific) questions
	def answerWh(self, wh, question, sentence):
		answer = ""
		answerLocs = []
		cQuestion = self.clean(question)
		cSentence = self.clean(sentence)
		if (wh == "who"):
			questionEnts = self.ner(cQuestion)
			sentenceEnts = self.ner(cSentence)
			for entNum in range(0,len(sentenceEnts)):
				if (sentenceEnts[entNum][1] == "PERSON"):
					answerLocs.append(entNum)
			answerLocs.append(-1)
			for locNum in range(0,len(answerLocs)-1):
				answer += sentenceEnts[answerLocs[locNum]][0]
				answer += " "
				if (answerLocs[locNum+1] - answerLocs[locNum] > 1):
					answer += "and"
					answer += " "
			if (answer == ""): answer = sentence # get original sentence
			return(answer)
		if (wh == "where"):
			questionEnts = self.ner(cQuestion)
			sentenceEnts = self.ner(cSentence)
			answer = "In "
			for entNum in range(0,len(sentenceEnts)):
				if (sentenceEnts[entNum][1] == "GPE"):
					answerLocs.append(entNum)
			answerLocs.append(-1)
			for locNum in range(0,len(answerLocs)-1):
				answer += sentenceEnts[answerLocs[locNum]][0]
				answer += " "
				if (answerLocs[locNum+1] - answerLocs[locNum] > 1):
					answer += "and"
					answer += " "
			if (answer == ""): answer = sentence # get original sentence
			return(answer)
		# time questions
		if (wh == "when"):
			words = cSentence.split()
			answer = ""
			inTimex = 0 # a tracker for if we are in a timex tagged phrase
			for wordNum in range(0,len(words)):
				if (words[wordNum] == "/TIMEX2"):
					inTimex = 0
				if (inTimex == 1):
					answer += words[wordNum]
					answer += " "
				if (words[wordNum] == "TIMEX2"):
					inTimex = 1
			year = re.compile("((?<=\s)\d{4}|^\d{4})")
			# make years sound more natural
			if (year.findall(answer)): answer = "In " + answer
			return(answer)
		# what question
		if (wh == "what" or wh == "which"):
			return(sentence) # return the whole best sentence

	def run(self):
		# answer all questions
		for i in range(0, len(self.questionList)):
			print("Question "+str(i)+": "+self.questionList[i])
			self.answerQuestion(self.questionList[i], self.sentenceList[i])

if __name__ == '__main__':
	AnsweringMachine(questionDoc=sys.argv[1], sentenceDoc=sys.argv[2]).run()