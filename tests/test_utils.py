from utils import *

def test_clean_str():

	test = 'This time the damage ... was estimated at $1 . 8 billion . This isJanuary 7 , 8 : 06 P. M.'
	print clean_str(test.lower())


def test_read_data():

	data_dir = '/Users/aditinair/Desktop/NLU-DL/Contextual-Conversational-Model/data/processed_en/'

	data_iter = read_data(data_dir)
	token_count = 0
	movie_count = 1
	for movie in data_iter:
		token_count += len(movie[1].split())
		movie_count += 1
	print 'Token count: ' + str(token_count)
	print 'Movie count: ' + str(movie_count)


def test_get_vocabulary():

	data_dir = '/Users/aditinair/Desktop/NLU-DL/Contextual-Conversational-Model/data/processed_en/'

	return get_vocabulary(1000, data_dir)


def test_corpus_to_idx_for_subset():

	#split by movie
	train, dev, test = shuffled_train_dev_split(data_dir, 0.01,0.10)

	#can compute corpus just for subsets
	train_corpus = corpus_to_idx(vocab, data_dir,subset=train)
	train_titles = train_corpus[0]
	train_tokens = train_corpus[1]

	return train_titles, train_tokens


def main():

	pass

if __name__ == '__main__':

	main()