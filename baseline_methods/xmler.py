import xml.etree.cElementTree as ET
from hazm import *
import os

tag_dict = {'Ne': 'NN', 'AJ': 'JJ', 'N': 'NN'}

tagger = POSTagger(model='postagger.model')
lemmatizer = Lemmatizer()


directory = './train3/'
paths = []
for file_name in os.listdir(directory):
	paths.append(file_name)

for file_name in paths:
	with open(directory + file_name) as file:
		data = file.read()

	temp = data.split('\n')
	sents = []
	for element in temp:
		sents.extend(sent_tokenize(element))

	sents = [tagger.tag(word_tokenize(sent)) for sent in sents]

	root = ET.Element("root")
	document = ET.SubElement(root, "document")
	sentences = ET.SubElement(document, "sentences")
	for i, sent in enumerate(sents):
		sentence = ET.SubElement(sentences, "sentence", id=str(i + 1))
		for j, wt_tuple in enumerate(sent):
			tokens = ET.SubElement(sentence, "tokens")
			token = ET.SubElement(tokens, "token", id=str(j + 1))
			word = ET.SubElement(token, "word").text=wt_tuple[0]
			lemma = ET.SubElement(token, "lemma").text=lemmatizer.lemmatize(wt_tuple[0])
			# CharacterOffsetBegin = ET.SubElement(token, "CharacterOffsetBegin").text=word
			# CharacterOffsetEnd = ET.SubElement(token, "CharacterOffsetEnd").text=word
			try:
				POS = ET.SubElement(token, "POS").text=tag_dict[wt_tuple[1]]
			except KeyError:
				POS = ET.SubElement(token, "POS").text=wt_tuple[1]

	tree = ET.ElementTree(root)
	tree.write('./xml_train3/' + file_name.replace('.txt', '') + '.xml')

