import os, re, sys
import numpy as np
import pickle
import modified_utils


def read_data(data_dir, subset=None):

	return modified_utils.read_data(data_dir, subset)


def clean_str(string):

	return modified_utils.clean_str(string)


def shuffled_train_dev_split(data_dir, train_split, dev_split):

	return modified_utils.shuffled_train_dev_split(data_dir, train_split, dev_split)


def create_cbow_data_files(input_dir, output_dir, train_split, dev_split, vocab_path, do_split=True):


	train_files, dev_files, test_files = shuffled_train_dev_split(input_dir, train_split, dev_split)

	#this should be the vocabulary with all the new UNKs
	vocabulary = pickle.load(open(vocab_path+'vocab.p','rb'))
	rev_vocabulary = pickle.load(open(vocab_path+'rev_vocab.p','rb'))
	vocab_size = len(vocabulary)

	#if you want to train the CBOW model
	if do_split:

		print 'Processing TRAIN files'
		train_output_filepath = output_dir+'cbow_TRAIN_'+str(vocab_size)+'.data'
		write_cbow_data(train_output_filepath,train_files,input_dir,vocabulary)

		print 'Processing DEV files'
		dev_output_filepath = output_dir+'cbow_DEV_'+str(vocab_size)+'.data'
		write_cbow_data(dev_output_filepath,dev_files,input_dir,vocabulary)

	else:

		print 'Processing TRAIN+DEV files'
		output_filepath = output_dir+'cbow_data_'+str(vocab_size)+'.data'
		cbow_files = train_files + dev_files

		write_cbow_data(output_filepath,cbow_files,input_dir,vocabulary)



def write_cbow_data_for_supervised(filepath, files_list, input_dir, vocabulary):

	with open(filepath, 'wb') as data_file:
			
		for f in files_list:

			print f

			with open(input_dir+f, 'rb') as current_movie_file:
				current_movie = current_movie_file.read().splitlines()

			for line in current_movie:

				sentence = line.split()

				if len(sentence) > 0:

					vocab_index = 0
					num_attempts = 0

					#try until we get a word in vocab that is not a special token (index 0 - 6)
					while vocab_index < 7 and num_attempts < 5:

						#pick a random word in the sentence
						random_word_index = np.random.randint(len(sentence))
						random_word = sentence[random_word_index]

						#make sure the random word is not something that would get tagged as proper noun
						#so it should not be an upper case word in the middle of the sentence
						if random_word[0].isupper() and random_word_index != 0:
							vocab_index = 0

						else:
							#get the index of that word in the vocabulary - set to 0=UNK if it's not in the vocabulary
							#make sure to lower case just in case... 
							vocab_index = vocabulary.get(random_word.lower(),0)

						num_attempts += 1

					#if we successfully picked a word in the vocabulary in 5 tries
					if vocab_index >= 7:

						new_sentence = sentence[:random_word_index] + sentence[random_word_index+1:]

						#let the labels be the indices in the vocabulary... this will be easier later on at training
						data_file.write(' '.join(word for word in new_sentence) + ' __label__' + str(vocab_index) + '\n' )


def write_cbow_data_for_unsupervised(filepath, files_list, input_dir, vocabulary):

	'''if we ever want to learn just word representations, 
	this is the data format we will need, adjusting for proper noun UNKS'''

	with open(filepath, 'wb') as data_file:

		for f in files_list:

			with open(input_dir+f, 'rb') as current_movie_file:
				current_movie = current_movie_file.read().splitlines()

			for line in current_movie:

				if len(line) > 0:

					sentence = modified_utils.combine_adjacent_uppers(line.split())
					unked_sentence = ['__PROPER_NOUN_UNK__' if token[0].isupper() else token.lower() for token in sentence]
					data_file.write(' '.join(token for token in unked_sentence))


def main():
	
	'''
	#first create the vocabulary with 100k vocabulary for special vocabulary
	vocabulary, rev_vocabulary = modified_utils.get_vocabulary(100000, 'corpus_token_counts.p')

	with open('./data/cbow_data_100000/vocab.p', 'wb') as vocab_pickle:
		pickle.dump(vocabulary, vocab_pickle)
	with open('./data/cbow_data_100000/rev_vocab.p', 'wb') as rev_vocab_pickle:
		pickle.dump(rev_vocabulary, rev_vocab_pickle)
	'''

	input_dir = './data/processed_en/'
	output_dir='./data/cbow_data_100000/'
	train_split = 0.8
	dev_split = 0.1
	vocab_path = output_dir
	create_cbow_data_files(input_dir,output_dir,train_split,dev_split,vocab_path,do_split=True)


if __name__ == '__main__':

	main()