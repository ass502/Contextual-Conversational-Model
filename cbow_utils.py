import os, re, sys
import numpy as np
import pickle

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

def corpus_to_sentences(data_dir, subset=None):

	titles = []
	corpus = []
	data_iter = read_data(data_dir, subset)

	for movie in data_iter:

		titles.append(movie[0])

		transcript = movie[1].splitlines()

		#for each movie, corpus contains 
		corpus.append(transcript)

	return titles, corpus

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

def create_cbow_data_files(input_dir, output_dir, train_split, dev_split, vocab_path):

	train_files, dev_files, test_files = shuffled_train_dev_split(input_dir, train_split, dev_split)
	cbow_files = list(train_files)+list(dev_files)

	vocabulary = pickle.load(open(vocab_path+'vocab.p',"rb"))
	rev_vocabulary = pickle.load(open(vocab_path+'rev_vocab.p',"rb"))

	vocab_size = len(vocabulary)

	titles, corpus = corpus_to_sentences(input_dir)

	#write train files
	with open(output_dir+"cbow_"+str(vocab_size)+".example",'wb') as example_f:
		with open(output_dir+"cbow_"+str(vocab_size)+".label",'wb') as label_f:
			for f in cbow_files:
				#find index of movie in titles
				for i in range(len(titles)):
					if titles[i] == f:
						break

				movie_text = corpus[i]

				#iterate through each sentence
				for s in movie_text:
					sentence = s.strip().split(" ")
					index = 0
					count = 0
					#until we get a word in vocab that is not a special token (indexes 0-3)
					while index < 4 and count < 5: #stop loop once we try 5 times
						random_word = np.random.randint(len(sentence))
						index = vocabulary.get(sentence[random_word],0)
						count += 1

					#if we successfully picked a word in the vocabulary in 5 tries
					if index >= 4:
						new_sentence = sentence[:random_word]+sentence[random_word+1:]
						#write the first line to the example file
						for i,w in enumerate(new_sentence):
							if i < len(new_sentence)-1:
								example_f.write(str(w)+" ")
							else:
								example_f.write(str(w)+'\n')

						label_f.write(str(index)+'\n')

def main():
	create_cbow_data_files("data/processed_en/","data/cbow_data/",.8,.1,"data/data_idx_files/2nd_run_100000/")

if __name__ == '__main__':

	main()