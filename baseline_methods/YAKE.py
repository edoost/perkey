import pke
import io


stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
    for line in stop_file:
        stoplist.append(line.replace('\n', ''))

for i in range(25000):
    input_file = './xml_test3/' + str(i) + '.xml'

    extractor = pke.unsupervised.YAKE(input_file=input_file, language=None)

    extractor.read_document(format='corenlp')

    extractor.candidate_selection(n=3, stoplist=stoplist)

    window = 2
    extractor.candidate_weighting(stoplist=stoplist, window=window, use_stems=False)

    threshold = 0.8
    keyphrases = extractor.get_n_best(n=50, threshold=threshold)

    with open('./results/yake_3/' + str(i) + '.txt', 'w+') as file:
        for key in keyphrases:
            file.write(key[0].encode('utf-8') + '\n')

