def _pairs_to_idx(sentence1, sentence2, vocabulary, cbow=None, replace_prob=1):

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
						idx1.append( token_to_idx(temp_token, vocabulary) )


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
			idx1.append( token_to_idx(token.lower(), vocabulary) )


	#now process the second sentence
	for token in tokens2:

		try:
			idx2.append( unk_assignments[token] )

		#raised if the current token is not in the unk assignments dictionary
		except KeyError:

			#it's possible this is a combination of a bunch of tokens
			temp_tokens = token.lower().split()
			for temp_token in temp_tokens:
				idx2.append( token_to_idx(temp_token, vocabulary) )

	return idx1, idx2
