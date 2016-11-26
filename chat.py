'''
At runtime, this script should accept checkpoint directories to figure out what model to load.

This script should accept command-line chats from the user. 
Then it will convert the user chats into one-hot encodings per the vocabulary. 
Then it will feed these chats into the encoder and receive one-hot encodings for the response.
It will print the response to the command line.
'''

import tensorflow as tf
import numpy as np
import utils
import pickle


tf.app.flags.DEFINE_boolean("interactive_chat", True, "Talk to a user!")
tf.app.flags.DEFINE_boolean("simulate_chat", False, "Simulate a chat by reading in static file.")
tf.app.flags.DEFINE_string("simulate_file", None, "File to read in for simulation")
tf.app.flags.DEFINE_string("checkpoint_dir", "./tmp", "Checkpoint directory.")

FLAGS = tf.app.flags.FLAGS


class Chat_Session(object):


	def __init__(self, vocabulary_file_path, rev_vocabulary_file_path):

		#keep track of tokens - not indexes - for legibility
		self.query_log = []
		self.response_log = []

		self.vocabulary = pickle.load(open(vocabulary_file_path, 'rb'))
		self.rev_vocabulary = pickle.load(open(rev_vocabulary_file_path, 'rb'))

		#load model and set the batch value to 1
		self.model = None


	def respond(self, idx_query):

		#this will require passing the query (and whatever else) to the decoder
		#then converting IDX to tokens

		response = '@#$!@#'
		return response


	def read_query(self):

		return raw_input('You: ')
		

	def chat(self):

		print ('HI! Let\'s chat!\n')

		while True:

			#read user query
			query = self.read_query()
			
			#transform query to idx
			idx_query = utils.sentence_to_idx(query, self.vocabulary)

			#compute response to user query
			response = self.respond(query)

			#print response
			print ('\nBOT: ' + response + '\n')

			#log user query and response
			self.query_log.append(query)
			self.response_log.append(response)


	def simulate(self, file_path):

		output_filepath = ''.join(str(i) for i in np.random.randint(9, size=10))

		with open(file_path, 'rb') as simulate_file:

			#read queries and transform to idx
			queries = simulate_file.read().splitlines()
			idx_queries = [utils.sentence_to_idx(sentence, self.vocabulary) for sentence in queries]

			for idx_query in idx_queries:

				#pass in query to model and decode response
				idx_response = self.respond(idx_query)

				#transform response to tokens
				response = ' '.join([reverse_vocabulary[i] for i in idx_response])

				#record query and response
				self.query_log.append(queries[idx])
				self.response_log.append(response)


	def save(self):
		
		filepath = ''.join(str(i) for i in np.random.randint(9, size=10))
		
		with open(filepath, 'wb') as log_file:
			
			#query log and response log should have the same length
			for idx, query in enumerate(self.query_log):

				#chats are in alternating USER QUERY - BOT RESPONSE order
				log_file.write('USER: ' + query + '\n')
				log_file.write('BOT: ' + self.response_log[idx] + '\n')

		print ('\nChat log saved to ' + filepath)


def main():

	vocabulary = './'
	reverse_vocabulary = './'
	session = Chat_Session(vocabulary, reverse_vocabulary)
	
	if FLAGS.simulate_chat:
		
		session.simulate(FLAGS.simulate_file)
		session.save()

	#this is the default 
	elif FLAGS.interactive_chat:

		try:
			session.chat()

		except EOFError, KeyboardInterrupt:

			#Give user option to save chat logs before exiting
			save = raw_input('\nType \'save\' to save a log of your chats. Type anything else to exit.\n')

			if save.strip().lower() == 'save':
				session.save()


if __name__ == '__main__':

	main()