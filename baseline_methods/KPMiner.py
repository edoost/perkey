import pke
import io


stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
    for line in stop_file:
        stoplist.append(line.replace('\n', ''))

df = pke.load_document_frequency_file(input_file='df_3_test.tsv.gz')

for i in range(25000):
    input_file = './test3/' + str(i) + '.txt'
    extractor = pke.unsupervised.KPMiner(input_file=input_file, language=None)

    extractor.read_document(format='raw')

    lasf = 3
    cutoff = 250
    extractor.candidate_selection(lasf=lasf, cutoff=cutoff, stoplist=stoplist)

    # df = pke.load_document_frequency_file(input_file='df.tsv.gz')

    alpha = 2.3
    sigma = 3.0
    try:
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)
    except:
        print(i)
    
    keyphrases = extractor.get_n_best(n=50)

    with open('./results/kp_3/' + str(i) + '.txt', 'w+') as file:
        for key in keyphrases:   
            file.write(key[0].encode('utf-8') + '\n')


