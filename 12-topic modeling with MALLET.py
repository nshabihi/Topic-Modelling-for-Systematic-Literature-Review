### Use Mallet for topic modeling
##run the following code only once:
#os.environ.update({'MALLET_HOME':r'C:/Mallet/'})
mallet_path = 'C:/Mallet/bin/mallet' # update this path

##use for loop the test results and plot them for more than one topic
#for i in range(20):  
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=6, id2word=id2word)


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# Select the model and print the topics
#Select the model from mallet of gensim: ldamallet or model_list[0]
optimal_model = ldamallet
model_topics = optimal_model.show_topics(formatted=True)
pprint(optimal_model.print_topics(num_words=30))
