'''
Aditi Nair
November 15

Data prep and utils for NCM

'''

import os, re, sys
import numpy as np
from collections import Counter


def read_data(data_dir, subset=None):

	#list all the text files in data_dir
	for f in os.listdir(data_dir):

		#all the scraped xml files end with .txt
		if f.endswith('.txt'):

			#this gives us the option of processing just certain movie titles at a time
			if (subset is not None and f in subset) or (subset is None):

				with open(data_dir+f, 'rb') as movie:

					yield f, clean_str(movie.read().lower())
	

def clean_str(string):

	'''
	The data is already cleaned quite well - but for our purposes we will need to replace times, dates and numbers
	'''

	#throw out -
	string = string.replace('-', ' ')

	#replace time formatting with _TIME_
	string = re.sub(r'[0-9]+ : [0-9]+ [a p]. m.', '_TIME_', string)

	#replace dates with _DATE_
	date_reg = r'(january|february|march|april|may|june|july|august|september|october|november|december) [0-9]{2} ,? [0-9]{0,4}'
	string = re.sub(date_reg, ' _DATE_', string)

	#replace dollar amounts with $__
	money_reg = r'\$[0-9]+ . [0-9]+'
	string = re.sub(money_reg, ' $__', string)

	#replace numbers with _NUM_ - allow spaces around regex so that there is room for tokens like 50s don't get abstracted
	num_reg = r'( [0-9]+)+'
	string = re.sub(num_reg, ' _NUM_', string)

	return string


def get_vocabulary(vocab_size, data_dir):

	corpus_counts = Counter({})

	data_iter = read_data(data_dir)
	i = 0
	for movie in data_iter:

		#for each movie, get the movie-level counts per token
		movie_token_counts = Counter(movie[1].split())

		#add to corpus level counts
		corpus_counts += movie_token_counts

		i += 1
		if i == 3:
			break

	#get the most common tokens in the corpus - subtract 4 for tokens below
	most_common_tokens = corpus_counts.most_common(vocab_size-4)

	#by default every vocabulary needs special tokens for unknown words, padding, end of sentence, start for decoder
	vocabulary = { '_UNK_': 0, '_PAD_': 1, '_EOS_':2, '_GO_': 3 }

	#add most common tokens to the vocabulary
	idx = max(vocabulary.values())
	for token in most_common_tokens:
		vocabulary[token[0]] = idx
		idx += 1

	return vocabulary


def corpus_to_idx(vocabulary, data_dir, subset):

	titles = []
	corpus = []
	data_iter = read_data(data_dir, subset)

	for movie in data_iter:

		titles.append(movie[0])

		transcript = movie[1].splitlines()

		idx_transcript = [sentence_to_idx(sentence,vocabulary) for sentence in transcript]

		#for each movie, corpus contains 
		corpus.append(idx_transcript)

	return titles, corpus


def sentence_to_idx(sentence, vocabulary):

	'''Sentence is a string - we convert it to the corresponding idx-es in the vocabulary.'''

	tokens = sentence.strip().split(' ')
	return [token_to_idx(token, vocabulary) for token in tokens]


def token_to_idx(token, vocabulary):

	try:
		return vocabulary[token]
	except KeyError:
		return vocabulary['_UNK_']


def shuffled_train_dev_split(data_dir, train_split, dev_split):

	movies = []
	for f in os.listdir(data_dir):
		if f.endswith('.txt'):
			movies.append(f)
	movies = np.array(movies)

	num_movies = len(movies)
	np.random.seed(0)
	shuffled_idx = np.random.permutation(len(movies))

	dev_idx = int(np.floor(train_split*num_movies))
	test_idx = int(np.floor((train_split+dev_split) * num_movies ) )

	#shuffled train test split by movie
	return movies[:dev_idx], movies[dev_idx:test_idx], movies[test_idx:] 


def batch_iter(data, batch_size, num_epochs):

	'''
	May not be necessary...
	'''

	data_size = len(data)
	num_partitions = int( np.floor(data_size/batch_size) )

	for epoch in num_epochs:

		c_idx = 0
		
		for partition in range(num_partitions):

			yield data[c_idx:c_idx+batch_size]
			c_idx += batch_size

		if num_partitions*batch_size != data_size:
			
			yield data[c_idx:]

		shuffled_idx = np.random.permutation(len(data))
		data = data[shuffled_idx]


def main():

	'''
	test = 'This time the damage ... was estimated at $1 . 8 billion . This isJanuary 7 , 8 : 06 P. M.'
	print clean_str(test.lower())
	'''

	data_dir = '/Users/aditinair/Desktop/NLU-DL/Contextual-Conversational-Model/data/processed_en/'

	data_iter = read_data(data_dir)
	token_count = 0
	movie_count = 1
	for movie in data_iter:
		token_count += len(movie[1].split())
		movie_count += 1
		print 'Token count: ' + str(token_count)
		print 'Movie count: ' + str(movie_count)

	'''
	vocab = get_vocabulary(1000, data_dir)

	#split by movie
	train, dev, test = shuffled_train_dev_split(data_dir, 0.01,0.10)
	
	#can compute corpus just for subsets
	train_corpus = corpus_to_idx(vocab, data_dir,subset=train)
	train_titles = train_corpus[0]
	train_tokens = train_corpus[1]
	'''


if __name__ == '__main__':

	main()