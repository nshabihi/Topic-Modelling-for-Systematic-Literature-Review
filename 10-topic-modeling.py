###TOPIC MODELLING

##dictionary and corpus before LDA
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

print("data_lemmatized: " + str(len(data_lemmatized)))
print("id2word: " + str(len(id2word)))
print(id2word[1])

# Create Corpus
texts = data_lemmatized
print("texts: " + str(len(texts)))

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
print("corpus: " + str(len(corpus)))

# Create the TF-IDF model
tfidf_model = TfidfModel(corpus, smartirs='ntc')
tfidf_vector = tfidf_model[corpus]

#show number of words in corpus
rejoined_docs = []
for item in data_lemmatized:
    words = item
    j = " ".join(words)
    rejoined_docs.append(j)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(rejoined_docs)

doc_term_matrix = X.todense()
dtm_df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
tdm_df = dtm_df.T
print("All courpus words: " + str(len(tdm_df)))



## LDA model
import logging
#for i in range(20,25): # if you want to check lda model for different number of topics
#Build LDA model
# after finding the optimal number of topics based on Coherence and Perplexity score,
#check different values for each parameter (e.g. alpha and beta) to optimize the model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6,  
                                           random_state=123,
                                           update_every=50,
                                           chunksize=50,
                                           passes=60, 
                                           alpha= 'auto', #0.1,
                                           per_word_topics=True,
                                           eta = 'auto',#0.6
                                           )

# Set up log to external log file
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Print the Keyword in the topics
pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
#print('number of topics: ', str(i+1))
print('Coherence Score: ', coherence_lda)
