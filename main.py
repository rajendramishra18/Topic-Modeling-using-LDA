import cPickle as cp
import gensim
from gensim import corpora

# load cleaned data containing all the documents
data = cp.load(open("exp.cp" , 'r'))

# create dictionary of all the terms indexed by their unique occurances
terms = corpora.Dictionary(data)

# create doc-term matrix
matrix = [terms.doc2bow(doc) for doc in data]

#create the LDA model
ldamodel = gensim.models.ldamodel.LdaModel
ldamodel = ldamodel(matrix, num_topics=20, id2word = terms, passes=50)

#print the results
print(ldamodel.print_topics(num_topics=20, num_words=10))
