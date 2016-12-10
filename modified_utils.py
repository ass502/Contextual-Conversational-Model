import pickle
import utils
import sys

'''
From utils borrow:
corpus_counts
clean_str
shuffled_train_dev_split

Create new:
read_data - don't necessarily want to lower case everything
get_vocabulary - need to incorporate CAPS_UNK_ID
pairs_to_idx - because the unk indices depend on the specific pair
combine_adjacent_uppers - so that "Paris France" is treated like a single token

create_train_test_files
'''

#standard default idxes for vocab
UNK_ID = 0
PAD_ID = 1
EOS_ID = 2
GO_ID = 3

#new placeholders for Proper Nouns - we will handle up to three in a sentence pair
CAPS_UNK_ID_1 = 4
CAPS_UNK_ID_2 = 5
CAPS_UNK_ID_3 = 6


'''-------------------------------BORROWED-------------------------------'''

def corpus_counts(data_dir):

	return utils.corpus_counts(data_dir)


def clean_str(string):

	return utils.clean_str(string)


def shuffled_train_dev_split(data_dir, train_split, dev_split):

	return utils.shuffled_train_dev_split(data_dir, train_split, dev_split)

'''-------------------------------NEW STUFF-------------------------------'''


def read_data(data_dir, subset=None):

	#list all the text files in data_dir
	for f in os.listdir(data_dir):

		#all the scraped xml files end with .txt
		if f.endswith('.txt'):

			#this gives us the option of processing just certain movie titles at a time
			if (subset is not None and f in subset) or (subset is None):

				with open(data_dir+f, 'rb') as movie:

					yield f, clean_str(movie.read())


def get_vocabulary(vocab_size, path_to_counts_pickle):

	#this pickle is generated in the script corpus_counts
	corpus_counts = pickle.load(open(path_to_counts_pickle, 'rb'))

	#get the most common tokens in the corpus - subtract 4 for tokens below
	most_common_tokens = corpus_counts.most_common(vocab_size-5)

	#by default every vocabulary needs special tokens for unknown words, padding, end of sentence, start for decoder
	vocabulary = { '_UNK_': UNK_ID, '_PAD_': PAD_ID, '_EOS_': EOS_ID, '_GO_': GO_ID, '_CAPS_UNK_ID_1_': CAPS_UNK_ID_1, 
		'_CAPS_UNK_ID_2_': CAPS_UNK_ID_2, '_CAPS_UNK_ID_3_': CAPS_UNK_ID_3 }

	#these happen to be in the correct order but it is not robust
	rev_vocabulary = ['_UNK_', '_PAD_', '_EOS_', '_GO_', '_CAPS_UNK_ID_1_', '_CAPS_UNK_ID_2_', 
		'_CAPS_UNK_ID_3_']

	#add most common tokens to the vocabulary
	idx = max(vocabulary.values()) + 1
	for token in most_common_tokens:
		vocabulary[token[0]] = idx
		rev_vocabulary.append(token[0])
		idx += 1

	return vocabulary, rev_vocabulary


def create_train_dev_test_files(input_dir, output_dir, train_split, dev_split, vocab_size, cbow=None, cbow_replace_prob=1):


	train_files, dev_files, test_files = shuffled_train_dev_split(input_dir, train_split, dev_split)
	vocabulary, rev_vocabulary = get_vocabulary(vocab_size, 'corpus_token_counts.p')

	#pickle vocabulary and rev vocabulary
	with open(output_dir+'vocab.p', 'wb') as vocab_pickle:
		pickle.dump(vocabulary, vocab_pickle)

	with open(output_dir+'rev_vocab.p', 'wb') as rev_vocab_pickle:
		pickle.dump(rev_vocabulary, rev_vocab_pickle)

	#write files for training set
	with open(output_dir+'train_'+str(vocab_size)+'.example', 'wb') as example_file:
		with open(output_dir+'train_'+str(vocab_size)+'.label', 'wb') as label_file:
			
			#go through all of the training set files
			for f in train_files:

				with open(input_dir+f, 'rb') as current_movie_file:
					current_movie = current_movie_file.read().splitlines()

				line_idx = 0
				while line_idx < len(current_movie):

					try:

						current_line = current_movie[line_idx]
						next_line = current_movie[line_idx+1]

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow, replace_prob=cbow_replace_prob)

						example_file.write(' '.join(str(i) for i in current_idx) + '\n')
						label_file.write(' '.join(str(i) for i in next_idx) + '\n')

					except IndexError:
						
						#there is no next pair!						
						pass

					line_idx += 1

				print 'Completed: ' + f

	#write files for dev set
	with open(output_dir+'dev_'+str(vocab_size)+'.example', 'wb') as example_file:
		with open(output_dir+'dev_'+str(vocab_size)+'.label', 'wb') as label_file:
			
			#go through all of the dev set files
			for f in dev_files:

				with open(input_dir+f, 'rb') as current_movie_file:
					current_movie = current_movie_file.read().splitlines()

				line_idx = 0
				while line_idx < len(current_movie):

					try:

						current_line = current_movie[line_idx]
						next_line = current_movie[line_idx+1]

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow, replace_prob=cbow_replace_prob)

						example_file.write(' '.join(str(i) for i in current_idx) + '\n')
						label_file.write(' '.join(str(i) for i in next_idx) + '\n')

					except IndexError:
						
						#there is no next pair!						
						pass

					line_idx += 1

				print 'Completed: ' + f


	#write files for test set
	with open(output_dir+'test_'+str(vocab_size)+'.example', 'wb') as example_file:
		with open(output_dir+'test_'+str(vocab_size)+'.label', 'wb') as label_file:
			
			#go through all of the test set files
			for f in test_files:

				with open(input_dir+f, 'rb') as current_movie_file:
					current_movie = current_movie_file.read().splitlines()

				line_idx = 0
				while line_idx < len(current_movie):

					try:

						current_line = current_movie[line_idx]
						next_line = current_movie[line_idx+1]

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow, replace_prob=cbow_replace_prob)

						example_file.write(' '.join(str(i) for i in current_idx) + '\n')
						label_file.write(' '.join(str(i) for i in next_idx) + '\n')

					except IndexError:
						
						#there is no next pair!						
						pass

					line_idx += 1

				print 'Completed: ' + f

	print 'Done!'


def pairs_to_idx(sentence1, sentence2, vocabulary, cbow=None, replace_prob=1):

	special_unk_assignments = {}

	sentence1 = sentence1.strip().split()
	sentence2 = sentence2.strip().split()

	#manually lower the first word in sentence1 and sentence2 so it won't get picked up by unk
	sentence1[0] = sentence1[0].lower()
	sentence2[0] = sentence2[0].lower()

	#combine adjoining tokens/words which start with upper case letters into a single token
	tokens1 = combine_adjacent_uppers( sentence1.strip().split() )
	tokens2 = combine_adjacent_uppers( sentence2.strip().split() )

	#set default vals for the idx lists
	sentence_idx1 = [-1]*len(tokens1)
	sentence_idx2 = [-1]*len(tokens2)


	#first set up the special unk correspondences
	for idx, token in enumerate(tokens1):

		#if the first token is upper, then we might either tag it with an existing UNK or give it a new UNK
		if token[0].isupper():

			#this means unk tokens have already begun to be assigned
			if len(unk_assignments) != 0:

				#first check if it matches a previous UNK
				if token in unk_assignments:
					sentence_idx1[idx] = unk_assignments[token] 

				#otherwise, check if you haven't run out of special unks
				#if you haven't, let the current token be a new special unk token
				elif max(unk_assignments.values()) != CAPS_UNK_ID_3:

					#set the current unk token
					curr_unk_token = max(unk_assignments.values()) + 1
					
					#update the unk assignments dictionary
					unk_assignments[token] = curr_unk_token

					#update the idx list
					sentence_idx1[idx] = curr_unk_token


			#if no unk tokens have been assigned yet
			else:

				#set the current unk token
				curr_unk_token = CAPS_UNK_ID_1

				#update the assignment dictionary
				unk_assignments[token] = curr_unk_token

				#update the idx list
				sentence_idx1[idx] = curr_unk_token


	#now modify sentence_idx2 according to the special unks from sentence_idx1
	for idx, token in enumerate(tokens2):
		if token in unk_assignments:
			sentence_idx2[idx] = unk_assignments[token]

	#if no cbow model is provided, convert remaining things to IDX as usual
	if cbow is None:

		for idx, vocab_idx in enumerate(sentence_idx1):
			if vocab_idx == -1:
				sentence_idx1[idx] = token_to_idx(tokens1[idx], vocabulary)

		for idx, vocab_idx in enumerate(sentence_idx2):
			if vocab_idx == -1:
				sentence_idx2[idx] = token_to_idx(tokens2[idx], vocabulary)


	#do cbow handling... 
	else:

		#now cast remaining sentence1 tokens to idx, keeping track of the cbow guesses
		cbow_guesses = {}
		cbow_not_guessing = []

		#first replace everything that you can with words in the vocabulary for sentence1
		for i, vocab_idx in enumerate(sentence_idx1):

			#if it's -1 it needs to be updated
			if vocab_idx == -1:

				try:
					new_vocab_idx = vocabulary[ tokens1[vocab_idx] ]
					sentence_idx1[i] = new_vocab_idx

				except KeyError:
					pass

		#next replace everything that you can with words in the vocabulary for sentence2
		for i, vocab_idx in enumerate(sentence_idx2):

			if vocab_idx == -1:

				try:
					new_vocab_idx = vocabulary[ tokens2[vocab_idx] ]
					sentence_idx2[i] = new_vocab_idx

				except KeyError:
					pass

		#now loop through the UNKs in sentence_idx1 and consider replacing them
		for i, vocab_idx in enumerate(sentence_idx1):

			if vocab_idx == -1:

				curr_token = tokens1[i]

				#see if the current word has been encountered before
				if curr_token in cbow_not_guessing:
					sentence_idx1[i] = vocabulary['_UNK_']

				elif curr_token in cbow_guesses:
					sentence_idx1[i] = cbow_guesses[curr_token]

				else:

					#do a weighted flip for whether to keep the guess or not
					if weighted_flip(replace_prob):

						#get the bag of words, as big as you can
						current_bag = [tokens1[idx] if val > CAPS_UNK_ID_3 for idx, val in enumerate(sentence_idx1)]
						bag_string = ' '.join(token.lower() for token in current_bag)
						unk_pred = model.predict([bag_string])

						cbow_guesses[ curr_token ] = unk_pred
						sentence_idx1[i] = vocabulary[unk_pred]

					else:

						cbow_not_guessing.append( curr_token )
						sentence_idx1[i] = vocabulary['_UNK_']


		#now loop through the UNKs in sentence_idx2 and consider replacing them
		for i, vocab_idx in enumerate(sentence_idx2):

			if vocab_idx == -1:

				curr_token = tokens2[i]

				#see if the current token has been encountered before
				if curr_token in cbow_not_guessing:
					sentence_idx2[i] = vocabulary['_UNK_']

				elif curr_token in cbow_guesses:
					sentence_idx2[i] = vocabulary[cbow_guesses[curr_token]]

				else:

					#do a weighted flip for whether to keep the guess or not
					if weighted_flip(replace_prob):

						#get the bag of words, as big as you can
						current_bag = [tokens2[idx] if val > CAPS_UNK_ID_3 for idx, val in enumerate(sentence_idx2)]
						bag_string = ' '.join(token.lower() for token in current_bag)
						unk_pred = model.predict([bag_string])

						cbow_guesses[ curr_token ] = unk_pred
						sentence_idx2[i] = vocabulary[unk_pred]

					else:

						cbow_not_guessing.append( curr_token )
						sentence_idx2[i] = vocabulary['_UNK_']

	return sentence_idx1, sentence_idx2


def combine_adjacent_uppers(tokens):

	new_tokens = []

	ptr = 0
	while ptr < len(tokens):
		
		try:
			if tokens[ptr][0].isupper():

				#see how many consecutive words are capitalized
				new_ptr = ptr
				while tokens[new_ptr+1][0].isupper():
					new_ptr += 1
				new_tokens.append( ' '.join(tok for tok in tokens[ptr:new_ptr+1]) )
				ptr = new_ptr+1

			#if it's not upper case, append it alone
			else:
				new_tokens.append( tokens[ptr] )
				ptr += 1

		#this happens if the pointer has overflow the tokens list
		except IndexError:
			try:
				new_tokens.append( tokens[ptr] )
				ptr += 1

			#this never happens.. 
			except IndexError:
				pass

	return new_tokens


def token_to_idx(token, vocabulary):

	try:
		return vocabulary[token]
	except KeyError:
		return vocabulary['_UNK_']


def get_cbow_guesses(tokens, model, replace_prob=1):

	for idx, token in enumerate(tokens):

		#if the token is UNK we consider replacing it
		if token == UNK_ID:
			
			#take as much info as you can from the current bag and use it for the prediction
			current_bag = [token for token in tokens if token > CAPS_UNK_ID_3]
			bag_string = ' '.join(str(token) for token in tokens)
			unk_pred = model.predict([bag_string])

			#do a weighted flip for whether to keep the guess or not
			if weighted_flip(replace_prob):
				tokens = tokens[:idx] + unk_pred + tokens[idx+1:]

	return tokens



def weighted_flip(prob):

	'''prob is the probability that it will return True'''

	return True if np.random.uniform() <= prob else False


def main():

	'''
	input_directory = './data/processed_en/'
	output_directory = './data/data_idx_files/small_model_10000_unks/'
	create_train_dev_test_files(input_directory, output_directory, train_split=0.8, dev_split=0.1, vocab_size=10000)
	'''

	#how often does this proper noun use case occur
	with open('./data/data_idx_files/small_model_10000_unks/train_10000.example', 'rb') as example_file:
		with open('./data/data_idx_files/small_model_10000_unks/train_10000.label', 'rb') as label_file:

			examples = example_file.read().splitlines()
			labels = label_file.read().splitlines()

			count = 0
			for idx in range(len(examples)):
				
				example = [int(i) for i in examples[idx].split()]
				label = [int(i) for i in labels[idx].split()]

				if CAPS_UNK_ID_1 in example and CAPS_UNK_ID_1 in label:
					count += 1
				
				if CAPS_UNK_ID_2 in example and CAPS_UNK_ID_2 in label:
					count += 1				
				
				if CAPS_UNK_ID_3 in example and CAPS_UNK_ID_3 in label:
					count += 1

	print count
	print len(examples)

if __name__ == '__main__':

	main()
