import pke
import io


stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
    for line in stop_file:
        stoplist.append(line.replace('\n', ''))

for i in range(25000):
    input_file = './xml_test2/' + str(i) + '.xml'

    extractor = pke.unsupervised.TopicRank(input_file=input_file, language=None)
    
    extractor.read_document(format='corenlp', use_lemmas=True)

    pos = set(['NN', 'JJ'])

    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    #extractor.grammar_selection(grammar="NP: {<NN.*>+<JJ.*>*}")
    #extractor.candidate_filtering(stoplist=stoplist)
    extractor.candidate_weighting(threshold=0.74, method='average')
    
    keyphrases = extractor.get_n_best(n=50, stemming=False)

    with open('./results/tr_2/' + str(i) + '.txt', 'w+') as file:
        for key in keyphrases:
            file.write(key[0].encode('utf-8') + '\n')
"""

extractor = pke.unsupervised.TopicRank(input_file='0.txt', language=None)
    
extractor.read_document(format='preprocessed', use_lemmas=False)

pos = set(['N', 'ADJ'])

extractor.candidate_selection(pos=pos, stoplist=stoplist)
    
extractor.candidate_weighting(threshold=0.74, method='average')
    
keyphrases = extractor.get_n_best(n=10, stemming=False)

with open('./TR_at_10/' + str(i) + '.txt', 'w+') as file:
    for key in keyphrases:
        file.write(key[0] + '\n')
"""
