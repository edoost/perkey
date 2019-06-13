import pke
import io


stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
    for line in stop_file:
        stoplist.append(line.replace('\n', ''))

for i in range(25000):
    input_file = './xml_test3/' + str(i) + '.xml'
    extractor = pke.unsupervised.MultipartiteRank(input_file=input_file, language=None)
    extractor.read_document(format='corenlp', use_lemmas=True)
    extractor.grammar_selection(grammar="NP: {<NN.*>+<JJ.*>*}")
    extractor.candidate_filtering(stoplist=stoplist)
    extractor.candidate_weighting(alpha=1.1, threshold=0.25, method='average')
    keyphrases = extractor.get_n_best(n=50)

    with open('./results/mr_3/' + str(i) + '.txt', 'w+') as file:
        for key in keyphrases:
            file.write(key[0].encode('utf-8') + '\n')

