##sentence to word

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False, min_len=3, max_len= 40))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
#print(data_words[30])

#### CKECK bigrams and 3-grams!
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=2, threshold=1) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  
#print(bigram)
       
#print(bigram.vocab)       
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
#print(trigram_mod[bigram_mod[data_words[0]]])

# show 3-grams
d = 0 
show_3grams = []
new_set  =trigram_mod[data_words]
for doc in new_set:
    for w in doc:
        if w.count("_") == 2:
            d+=1
            show_3grams.append(w)
show_3grams_dict = {i:(show_3grams.count(i)+1) for i in show_3grams} 
show_3grams_dict_sorted = {k: v for k, v in sorted(show_3grams_dict.items(), key=lambda item: item[1])}

#print(len(show_3grams_dict_sorted))
keys = []
for key in show_3grams_dict_sorted:
    keys.append(key)
Print("3grams:")
print(show_3grams_dict_sorted)


# show bigrams
d = 0 
show_bigrams = []
new_set  =bigram_mod[data_words]
for doc in new_set:
    for w in doc:
        if '_' in w:
            d+=1
            show_bigrams.append(w)
show_bigrams_dict = {i:(show_bigrams.count(i)+1) for i in show_bigrams} 
show_bigrams_dict_sorted = {k: v for k, v in sorted(show_bigrams_dict.items(), key=lambda item: item[1])}

#print(len(show_bigrams_dict_sorted))
keys = []
for key in show_bigrams_dict_sorted:
    keys.append(key)
print("bi-garms:")
print(show_bigrams_dict_sorted)
