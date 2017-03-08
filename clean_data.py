from __future__ import unicode_literals, print_function
import cPickle as cp
import csv
import numpy as np
import random
import collections
from itertools import groupby
from sets import Set
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from spacy.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models


def parse_html(html_doc):
	clean_text = []
	for each in html_doc:
		soup = BeautifulSoup(each.decode("utf-8"), 'html.parser')
		clean_text.append(soup.get_text())
	return clean_text

def tokenize(doc , model):
	doc_list = []
	for each in doc:
		text = model(each.encode("utf-8").decode("utf-8"))
		pos_doc = []
		for sent in text.sents:
			for word in sent:
				tag = word.tag_
				pos = word.pos_
				if 'NN' in tag or 'ADJ' in pos or 'ADV' in pos or 'VERB' in pos:
					#print(tag)
					pos_doc.append(word.lemma_.lower())
		doc_list.append(pos_doc)
	return doc_list

def remove_stopwords(stop, doc_list):
	cleaned_doc = []
	for doc in doc_list:
		temp = []
		for each in doc:
			if each not in stop:
				temp.append(each)
		cleaned_doc.append(temp)
						
	return cleaned_doc


def list_to_string(doc):
	complete_topic = []
	for i in range(0,len(doc)):
		para=doc[i]
		string = ' '
		for j in range(0 , len(para)):
			sent = para[j]
			for k in range(0 , len(sent)):
				#print(sent[k])
				string = string.join(sent[k]["word"])
	
		complete_topic.append(string)
	return complete_topic	

def compute_tfidf(doc):
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
	tfidf_matrix =  tf.fit_transform(doc)
	feature_names = tf.get_feature_names()
	print(feature_names)
	print(tfidf_matrix)
	return tfidf_matrix , feature_names

def is_capital(word):
	return int(not word.isupper() or word.islower())

def is_allcapital(word):
	return int(word.isupper())

def contains_digit(word):
	return int(word.isalnum())	

def contains_special_char(word):
	return int(not word.isalnum() or word.isdigit() or word.isalpha())

def is_noun(word):
	tag = word["tag"]
	return int("NN" in tag) 

def is_adjective(word):
	tag = word["tag"]
	return int("JJ" in tag)

def noun_type(word):
	tag = word["tag"]
	label = [0,0,0,0]
	if tag =="NN":
		label[0] = 1
	elif tag == "NNP":
		label[1] = 1
	elif tag == "NNPS":
		label[2] = 1
	elif tag == "NNS":
		label[3] = 1
	return label

def adjective_type(word):
	tag = word["tag"]
	label = [0,0,0]
	if tag =="JJ":
		label[0] = 1
	elif tag == "JJR":
		label[1] = 1
	elif tag == "JJS":
		label[2] = 1
	return label

def centrality_with_topic(topic , word):
	return int(word in topic)

def similarity_with_topic(topic , text , model):
	return model.similarity(topic , text)


''' Function to read tsv '''
def read_csv(file_name):
	topic = []
	text = []
	phrase = []
	with open(file_name , "r") as data_file:
		data = csv.reader(data_file , delimiter = str(','))
		for row in data:
			topic.append(row[1])
			text.append(row[2])
			phrase.append(row[3])
	return topic , text , phrase


def prepare_vectors(pos ,topic ,text,  tfidf_matrix , feature_names,word2vec , phrase):
	word_embedding = []
	word_label = []
	for i in range(len(pos)):
		print(i)
		para = pos[i]
		#print(phrase[i])
		tag = phrase[i]
		tag = tag.split(" ")
		#print(tag)
		word_vector_text = []
		#print(text[i])
		#print(topic[i])
		#sim_score = similarity_with_topic(word2vec , text[i] , topic[i])
		term = tfidf_matrix[i]
		episode = term.todense().tolist()[0]
		phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
		sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
		dictionary = {}
		p = [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores]
		phrase_list = []
		for phrase_word,score in p:
			dictionary[phrase_word] = score
			phrase_list.append(phrase_word)
		#print(dictionary)
		label_list = []
		for j in range(len(para)):
			sent = para[j]
			sent_mat = []
			label_mat = []
			for words in sent:
				word = str(words["word"])
				#print(word)
				if word in tag:
					print(word)
					label = [0,1]
				else:
					label = [1,0]
				if word.lower() in word2vec:
					vector = list(word2vec[word.lower()])
				else:
					vector = list(np.zeros(50))
				if word in phrase_list:
					vector.append(dictionary[word])
				else:
					vector.append(0)
				vector.append(is_capital(word))
				vector.append(is_allcapital(word))
				vector.append(contains_digit(word))
				vector.append(contains_special_char(word))
				vector.append(is_noun(words))
				vector.append(is_adjective(words))
				vector = vector + noun_type(words)
				vector = vector + adjective_type(words)
				vector.append(centrality_with_topic(topic , word))
		
				#vector.append(sim_score)
				sent_mat.append(vector)
				label_mat.append(label)
			label_list.append(label_mat)
			word_vector_text.append(sent_mat)
		word_embedding.append(word_vector_text)
		word_label.append(label_list)
	cp.dump(word_embedding , open("train_X.cp","w"))
	cp.dump(word_label , open("train_Y.cp" , "w"))

	return word_embedding , word_label


text = []
topic = []

with open("biology.csv" , 'r') as datafile:
	data = csv.reader(datafile , delimiter = str(','))
	for row in data:
		text.append(row[2])
		topic.append(row[1])

# Extract raw text parsing the HTML
topic = parse_html(topic)

text = parse_html(text)

merged_text = []
for i in range(0,len(topic)):
	merged_text.append(topic[i]+ " " +text[i])



# Perform tokenization and POS tagging
model = English()
sent_list = tokenize(merged_text , model)


# Remove stopwords
stop = set(stopwords.words('english'))
doc_cleaned = remove_stopwords(stop, sent_list)

cp.dump(doc_cleaned , open("exp.cp" , 'w'))

