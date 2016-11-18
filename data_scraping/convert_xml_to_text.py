from xml.etree import cElementTree as ET
import os
import sys

#track counts 
sentence_count = 0
token_count = 0

with open("open_subtitles_data.txt","wb") as out_f:
	#recursively walk through OpenSubtitles directory
	for root, dirs, files in os.walk('en'):
		for name in files:
			#get filename of current file
			filename = os.path.join(root,name)
			
			#parse only if it is an .xml file
			if filename[-4:] == '.xml':
				#xml parser
				document = ET.parse(filename)
				tree = document.getroot()

				#each child of the document tree is a sentence
				for c,child in enumerate(tree):
					sentence = ""
					for i in range(len(child)):
						#words of sentences have a w tag, ignore other tags
						if child[i].tag == 'w':
							sentence += child[i].text
							sentence += " "
							token_count += 1

					try:
						#remove space at end of sentence
						if sentence[-1] == " ":
							sentence = sentence[:-1]
						sentence_count += 1
					except IndexError: #if sentence is empty
						pass

					out_f.write(sentence.encode('utf-8')+'\n')
				out_f.write('\n')
				

print "Sentence count: " + str(sentence_count)
print "Token count: " + str(token_count)
	