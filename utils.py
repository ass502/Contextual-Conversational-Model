'''
Aditi Nair
November 15

Data prep and utils for NCM

'''

import os, re, sys
import numpy as np
from collections import Counter
import pickle

UNK_ID = 0
PAD_ID = 1
EOS_ID = 2
GO_ID = 3

def read_data(data_dir, subset=None):

	#list all the text files in data_dir
	for f in os.listdir(data_dir):

		#all the scraped xml files end with .txt
		if f.endswith('.txt'):

			#this gives us the option of processing just certain movie titles at a time
			if (subset is not None and f in subset) or (subset is None):

				with open(data_dir+f, 'rb') as movie:

					yield f, clean_str(movie.read().lower())
	

def corpus_counts(data_dir):

	corpus_counts = Counter({})

	data_iter = read_data(data_dir)
	for movie in data_iter:

		#for each movie, get the movie-level counts per token
		movie_token_counts = Counter(movie[1].split())

		#add to corpus level counts
		corpus_counts += movie_token_counts

		print ('Counted tokens for ' + movie[0])

	pickle.dump(corpus_counts,open('corpus_token_counts.p', 'wb'))



def clean_str(string):

	'''
	The data is already cleaned quite well - but for our purposes we will need to replace times, dates and numbers
	'''

	#throw out - and other chars
	char_reg = r'[- \] \[ ]'
	string = re.sub(char_reg, ' ', string)

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


def get_vocabulary(vocab_size, path_to_counts_pickle):

	#this pickle is generated in the script corpus_counts
	corpus_counts = pickle.load(open(path_to_counts_pickle, 'rb'))

	#get the most common tokens in the corpus - subtract 4 for tokens below
	most_common_tokens = corpus_counts.most_common(vocab_size-4)

	#by default every vocabulary needs special tokens for unknown words, padding, end of sentence, start for decoder
	vocabulary = { '_UNK_': UNK_ID, '_PAD_': PAD_ID, '_EOS_': EOS_ID, '_GO_': GO_ID }

	#these happen to be in the correct order but it is not robust
	rev_vocabulary = ['_UNK_', '_PAD_', '_EOS_', '_GO_']

	#add most common tokens to the vocabulary
	idx = max(vocabulary.values()) + 1
	for token in most_common_tokens:
		vocabulary[token[0]] = idx
		rev_vocabulary.append(token[0])
		idx += 1

	return vocabulary, rev_vocabulary


def corpus_to_idx(vocabulary, data_dir, subset=None):

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

	#shuffle!
	np.random.seed(0)
	shuffled_idx = np.random.permutation(len(movies))
	movies = movies[shuffled_idx]

	dev_idx = int(np.floor(train_split*num_movies))
	test_idx = int(np.floor((train_split+dev_split) * num_movies ) )

	#shuffled train test split by movie
	return movies[:dev_idx], movies[dev_idx:test_idx], movies[test_idx:] 


def create_train_dev_test_files(input_dir, output_dir, train_split, dev_split, vocab_size):

	train_files, dev_files, test_files = shuffled_train_dev_split(input_dir, train_split, dev_split)

	vocabulary,rev_vocabulary = get_vocabulary(vocab_size, 'corpus_token_counts.p')

	titles, corpus = corpus_to_idx(vocabulary, input_dir)

	#pickle vocabulary and rev vocabulary
	with open(output_dir+'vocab.p', 'wb') as vocab_pickle:
		pickle.dump(vocabulary, vocab_pickle)

	with open(output_dir+'rev_vocab.p', 'wb') as rev_vocab_pickle:
		pickle.dump(rev_vocabulary, rev_vocab_pickle)

	#write train files
	with open(output_dir+"train_"+str(vocab_size)+".example",'wb') as example_f:
		with open(output_dir+"train_"+str(vocab_size)+".label",'wb') as label_f:
			for f in train_files:
				#find index of movie in titles
				for i in range(len(titles)):
					if titles[i] == f:
						break

				movie_text = corpus[i]

				#iterate through each sentence
				for s in range(len(movie_text)-1):
					#write the first line to the example file
					for w in range(len(movie_text[s])):
						if w < len(movie_text[s])-1:
							example_f.write(str(movie_text[s][w])+" ")
						else:
							example_f.write(str(movie_text[s][w]))
					example_f.write("\n")

					#write the second line to the label file
					for w in range(len(movie_text[s+1])):
						if w < len(movie_text[s+1])-1:
							label_f.write(str(movie_text[s+1][w])+" ")
						else:
							label_f.write(str(movie_text[s+1][w]))
					label_f.write("\n")

	#write dev files
	with open(output_dir+"dev_"+str(vocab_size)+".example",'wb') as example_f:
		with open(output_dir+"dev_"+str(vocab_size)+".label",'wb') as label_f:
			for f in dev_files:
				#find index of movie in titles
				for i in range(len(titles)):
					if titles[i] == f:
						break

				movie_text = corpus[i]

				#iterate through each sentence
				for s in range(len(movie_text)-1):
					#write the first line to the example file
					for w in range(len(movie_text[s])):
						if w < len(movie_text[s])-1:
							example_f.write(str(movie_text[s][w])+" ")
						else:
							example_f.write(str(movie_text[s][w]))
					example_f.write("\n")

					#write the second line to the label file
					for w in range(len(movie_text[s+1])):
						if w < len(movie_text[s+1])-1:
							label_f.write(str(movie_text[s+1][w])+" ")
						else:
							label_f.write(str(movie_text[s+1][w]))
					label_f.write("\n")

	#write test files
	with open(output_dir+"test_"+str(vocab_size)+".example",'wb') as example_f:
		with open(output_dir+"test_"+str(vocab_size)+".label",'wb') as label_f:
			for f in test_files:
				#find index of movie in titles
				for i in range(len(titles)):
					if titles[i] == f:
						break

				movie_text = corpus[i]

				#iterate through each sentence
				for s in range(len(movie_text)-1):
					#write the first line to the example file
					for w in range(len(movie_text[s])):
						if w < len(movie_text[s])-1:
							example_f.write(str(movie_text[s][w])+" ")
						else:
							example_f.write(str(movie_text[s][w]))
					example_f.write("\n")

					#write the second line to the label file
					for w in range(len(movie_text[s+1])):
						if w < len(movie_text[s+1])-1:
							label_f.write(str(movie_text[s+1][w])+" ")
						else:
							label_f.write(str(movie_text[s+1][w]))
					label_f.write("\n")
	


def batch_iter(data, batch_size, num_epochs):

	'''
	DON'T NEED THIS
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

	#corpus_counts("data/processed_en/")

	create_train_dev_test_files("data/processed_en/", "data/data_idx_files/small_model_10000/", .8, .1, 10000)


if __name__ == '__main__':

	main()