import pke
import io


stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
    for line in stop_file:
        stoplist.append(line.replace('\n', ''))

for i in range(25000):
    input_file = './xml_test2/' + str(i) + '.xml'
    
    extractor = pke.unsupervised.SingleRank(input_file=input_file, language=None)

    extractor.read_document(format='corenlp')

    pos = set(['NN', 'JJ'])

    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    
    #extractor.grammar_selection(grammar="NP: {<NN.*>+<JJ.*>*}")
    #extractor.candidate_filtering(stoplist=stoplist)

    extractor.candidate_weighting(window=10)

    keyphrases = extractor.get_n_best(n=50)
   
    with open('./results/sr_2/' + str(i) + '.txt', 'w+') as file:
        for key in keyphrases:
            file.write(key[0].encode('utf-8') + '\n')


