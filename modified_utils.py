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


def create_train_dev_test_files(input_dir, output_dir, train_split, dev_split, vocab_size, cbow=False):


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

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow=False)

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

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow=False)

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

						current_idx, next_idx = pairs_to_idx(current_line, next_line, vocabulary, cbow=False)

						example_file.write(' '.join(str(i) for i in current_idx) + '\n')
						label_file.write(' '.join(str(i) for i in next_idx) + '\n')

					except IndexError:
						
						#there is no next pair!						
						pass

					line_idx += 1

				print 'Completed: ' + f

	print 'Done!'


def pairs_to_idx(sentence1, sentence2, vocabulary, cbow=False):

	unk_assignments = {}

	tokens1 = combine_adjacent_uppers( sentence1.strip().split() )
	tokens2 = combine_adjacent_uppers( sentence2.strip().split() )

	#manually lowercase the first word in tokens1 so it won't get picked up by unk
	#don't bother with tokens2 since tokens1 determines the special unks
	tokens1[0] = tokens1[0].lower()

	idx1 = []
	idx2 = []

	for token in tokens1:

		#if the first token is upper, then we might either tag it with an existing UNK or give it a new unk
		if token[0].isupper():

			#this means unk tokens have already begun to be assigned
			if len(unk_assignments) != 0:

				#first check if it matches a previous UNK
				if token in unk_assignments:
					idx1.append( unk_assignments[token] )


				#otherwise, if you have already run out of special unks, default to the usual method
				elif max(unk_assignments.values()) == CAPS_UNK_ID_3:

					#split them back up!
					temp_tokens = token.lower().split()
					for temp_token in temp_tokens:
						idx1.append( token_to_idx(temp_token, vocabulary, cbow) )


				#otherwise, let the current token be a new special unk token
				else:

					#set the current unk token
					curr_unk_token = max(unk_assignments.values()) + 1
					
					#update the unk assignments dictionary
					unk_assignments[token] = curr_unk_token

					#update the idx list
					idx1.append( curr_unk_token )

			#if no unk tokens have been assigned yet
			else:

				#set the current unk token
				curr_unk_token = CAPS_UNK_ID_1

				#update the assignment dictionary
				unk_assignments[token] = curr_unk_token

				#update the idx list
				idx1.append( curr_unk_token )

		#if the current token is fully lowercase, do the usual method
		else:

			#note this does not implement edit threshold - and lowercase it!
			idx1.append( token_to_idx(token.lower(), vocabulary, cbow) )


	#now process the second sentence
	for token in tokens2:

		try:
			idx2.append( unk_assignments[token] )

		#raised if the current token is not in the unk assignments dictionary
		except KeyError:

			#it's possible this is a combination of a bunch of tokens
			temp_tokens = token.lower().split()
			for temp_token in temp_tokens:
				idx2.append( token_to_idx(temp_token, vocabulary, cbow) )

	return idx1, idx2


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


def token_to_idx(token, vocabulary, cbow=False):

	try:
		return vocabulary[token]
	except KeyError:
		if cbow is False:
			return vocabulary['_UNK_']
		else:
			return None

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
