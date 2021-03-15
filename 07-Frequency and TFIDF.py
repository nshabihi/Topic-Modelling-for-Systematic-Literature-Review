####to CKECK word ranks for deciding which words to keep/delete!
## word ranking
rejoined_docs = []
for item in data_lemmatized:
    words = item
    j = " ".join(words)
    rejoined_docs.append(j)

#create tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(rejoined_docs)
terms = vectorizer.get_feature_names()
sums = X.sum(axis=0)
a = []
for col, term in enumerate(terms):
    a.append((term, sums[0,col] ))
    
ranking = pd.DataFrame(a, columns=['term','rank'])
ranked_terms_tfidf = (ranking.sort_values('rank', ascending=False))
#print("all words:  " + str(len(ranked_terms_tfidf)))
print("first 50 words based on TFIDF ranks :  ")
print(ranked_terms_tfidf[0:49])
print("//////////")
#print(ranked_terms_tfidf.iloc[0 ,0])
#print(ranked_terms_tfidf.iloc[0,1])
#print("//////////")
list_tfidf = []
list_tfidf2 = []
for i in range(len(ranked_terms_tfidf)):
    if 0<ranked_terms_tfidf.iloc[i,1]<0.25:
        list_tfidf.append(ranked_terms_tfidf.iloc[i,0])
    elif 0.49<ranked_terms_tfidf.iloc[i,1]<1.01:
        list_tfidf2.append(ranked_terms_tfidf.iloc[i,0])
print("TFIDF lists 0 - 0.25:")
print(list_tfidf)
print("TFIDF lists 0.49 - 1:")
print(list_tfidf2)
print("////////////")

import matplotlib.pyplot as plt
plt.plot((ranked_terms_tfidf.iloc[: ,0]),(ranked_terms_tfidf.iloc[: ,1]))
plt.title('TFIDF')
plt.xlabel('xAxis name')
plt.ylabel('yAxis name')
plt.show()


#create dtm
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(rejoined_docs)
terms = vectorizer.get_feature_names()


sums = X.sum(axis=0)
a = []
for col, term in enumerate(terms):
    a.append( (term, sums[0,col] ))
    
ranking = pd.DataFrame(a, columns=['term','rank'])
ranked_terms_freq = ranking.sort_values('rank', ascending=False)



#print("all: " + str(len(ranked_terms_freq)))
#delete low_frequent words

def word_removal(texts):
    texts = [[word for word in doc if word not in del_list] for doc in texts]  
    return texts
del_list = []
my_list = []
my_list1 = []
for i in range(len(ranked_terms_freq)):
    if 0<ranked_terms_freq.iloc[i]['rank'] < 10:
        del_list.append(ranked_terms_freq.iloc[i]['term'])
    elif  9<ranked_terms_freq.iloc[i]['rank']<100:
        my_list.append(ranked_terms_freq.iloc[i]['term'])
    elif  99<ranked_terms_freq.iloc[i]['rank']:
        my_list1.append(ranked_terms_freq.iloc[i]['term'])
        
#print("//////////count list-1:   "+str(len(del_list)))
#print(del_list)
#print("//////////count list-2:  "+str(len(my_list)))
#print(my_list)
#print("//////////count list-3:  "+str(len(my_list1)))
#print(my_list1)
#data_lemmatized = word_removal(data_lemmatized)
print("first 50 words based on frequency ranks :  ")
print(ranked_terms_freq[0:49])

#frequency plot
plt.plot((ranked_terms_freq.iloc[: ,0]),(ranked_terms_freq.iloc[: ,1]))
plt.title('frequency')
plt.xlabel('xAxis name')
plt.ylabel('yAxis name')
plt.show()



#print("++++++++++")
n = 0
for i in range(len(ranked_terms_freq)):
    if ("_" in ranked_terms_freq.iloc[i,0]):
        #print(ranked_terms_freq.iloc[i,0])
        n += 1
#print (n)

#print("++++++++++")
word_list = []
for i in range(len(ranked_terms_freq)):
    word_list.append(ranked_terms_freq.iloc[i,0])
#print (word_list)


#RUN if want to check the number of keywords in a doc
n = 0
for i in range(len(data_lemmatized)):
    if len(data_lemmatized[i]) < 4:
        #print(str(i)+ " :   "+ str(len(data_lemmatized[i])))
        #print(data_lemmatized[i])
        n += 1
        
#print(n)
