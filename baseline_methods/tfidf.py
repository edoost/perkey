import pke
import io

stoplist = []
with io.open('stop.txt', encoding='utf-8') as stop_file:
  for line in stop_file:
    stoplist.append(line.replace('\n', ''))

#pke.utils.compute_document_frequency('./test2', 'df_2_test.tsv.gz', format='raw', extension='txt', use_lemmas=False, stemmer=None, stoplist=stoplist, delimiter='\t', n=3)

for i in range(25000):
  input_file = './test2/' + str(i) + '.txt'

  # 1. create a TfIdf extractor.
  extractor = pke.unsupervised.TfIdf(input_file=input_file)

  # 2. load the content of the document.
  extractor.read_document(format='raw', use_lemmas=False, stemmer=None, sep='/')

  # 3. select {1-3}-grams not containing punctuation marks as candidates.
  n = 3

  extractor.candidate_selection(n=n, stoplist=stoplist)

  # 4. weight the candidates using a `tf` x `idf`
  df = pke.load_document_frequency_file(input_file='df_2_test.tsv.gz')
  extractor.candidate_weighting(df=df)

  # 5. get the 10-highest scored candidates as keyphrases
  keyphrases = extractor.get_n_best(n=50)

  with open('./results/tfidf_2/' + str(i) + '.txt', 'w+') as file:
    for key in keyphrases:
      file.write(key[0].encode('utf-8') + '\n')

