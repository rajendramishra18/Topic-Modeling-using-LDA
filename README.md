# Topic-Modeling-using-LDA
Topic Modeling using LDA with the help of gensim and spacy.

Given below are some of the terms that are extracted from the given documents. Each entry in the list represents a topic.

topic = [(0, u'0.073*"plant" + 0.034*"energy" + 0.020*"atp" + 0.016*"glucose" + 0.016*"produce"'), (1, u'0.066*"gene" + 0.039*"sequence" + 0.019*"genome" + 0.011*"expression" + 0.010*"find"'), (2, u'0.088*"blood" + 0.022*"pressure" + 0.013*"volume" + 0.012*"red" + 0.009*"mm"'), (3, u'0.093*"protein" + 0.032*"acid" + 0.020*"structure" + 0.020*"amino" + 0.014*"molecule"'), (4, u'0.018*"specie" + 0.018*"human" + 0.015*"would" + 0.014*"animal" + 0.010*"question"'), (5, u'0.021*"see" + 0.020*"eye" + 0.014*"light" + 0.014*"find" + 0.012*"\'"'), (6, u'0.043*"tree" + 0.026*"bird" + 0.022*"sleep" + 0.021*"fish" + 0.010*"specie"'), (7, u'0.015*"human" + 0.013*"time" + 0.013*"rate" + 0.012*"age" + 0.010*"grow"'), (8, u'0.042*"food" + 0.036*"eat" + 0.011*"diet" + 0.010*"milk" + 0.009*"predator"'), (9, u'0.032*"population" + 0.030*"mutation" + 0.025*"male" + 0.022*"female" + 0.013*"sex"'), (10, u'0.024*"body" + 0.017*"ant" + 0.016*"fat" + 0.014*"weight" + 0.014*"muscle"'), (11, u'0.226*"cell" + 0.021*"cancer" + 0.018*"tissue" + 0.011*"type" + 0.009*"stem"'), (12, u'0.020*"use" + 0.018*"find" + 0.017*"would" + 0.012*"datum" + 0.010*"know"'), (13, u'0.026*"brain" + 0.026*"neuron" + 0.020*"signal" + 0.017*"receptor" + 0.017*"muscle"'), (14, u'0.108*"dna" + 0.024*"enzyme" + 0.021*"strand" + 0.017*"rna" + 0.014*"primer"'), (15, u'0.026*"membrane" + 0.022*"water" + 0.019*"concentration" + 0.013*"leaf" + 0.012*"potential"'), (16, u'0.034*"use" + 0.015*"plasmid" + 0.014*"sample" + 0.010*"rna" + 0.010*"gel"'), (17, u'0.023*"would" + 0.016*"get" + 0.015*"body" + 0.011*"make" + 0.010*"\'"'), (18, u'0.052*"chromosome" + 0.022*"allele" + 0.021*"b" + 0.019*"x" + 0.018*"c"'), (19, u'0.035*"bacteria" + 0.027*"virus" + 0.024*"disease" + 0.013*"immune" + 0.013*"system"')]

For example, topic[0] contains the terms like plant, energy, atp, glucose, produce. So the most appropriate topic for these terms is "Photosynthesis".

topic[2] contains the terms like blood, pressure, volume, red, mm. So the most appropriate topic for these terms is "Blood Pressure".

Cleaning of the text document includes Stop word removal , lemmatization and POS tagging based filtering.
Spacy has been used to tokenize the sentence into words, finding stems for the words, tagging the POS for each word and then filtering out the words which has their POS labels other than:
'NN' in (tag) or 'ADJ' in (pos) or 'ADV' in (pos) or 'VERB' in (pos)
