### Outlier Removal
##find the treshold for cosine similarity to delete outliers:

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
data_M = X.todense()
data_df = pd.DataFrame(data_M, columns=vectorizer.get_feature_names())
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine = (cosine_similarity(data_df, data_df))

mean_cosine = cosine.mean(1)
similarity_counter = []
n = 0
temp_doc_list = []
doc_list = []
for j in range(20):
    for i in range(len(mean_cosine)):
        if ( mean_cosine[i] < (j+1) * 0.025):
            n += 1
            temp_doc_list.append(str(i))
    similarity_counter.append(n)
    doc_list.append(temp_doc_list)
    temp_doc_list = []
    n=0
print(len(data))
print("Number of Outliers Found: ")
print(similarity_counter)

import matplotlib.pyplot as plt 
import numpy as np 
  
# define plot values
x = []
for j in range(20):
    x.append(j * 0.025)
y = similarity_counter
plt.xlabel('MeanCosine Treshold')
plt.ylabel('Number of outliers found')
plt.plot(x, y)  
plt.show() 

for i in range(len(doc_list)):
    print("outlier documents when cosine treshold is : " +str(i*0.025))
    print(doc_list[i])
    
    
####Result from finding Treshold: we selected 0.075 and deleted documents that their 
#mean similarity to other docs is less than 0.075, we also manually checked 
#documents with 0.1 similarity and add some more docs to delete

#### delete less similar or unrelevant docs (outliers): it works based on the above cosine similarity
del_list = [142, 153, 422, 140,154, 211,374, 416, 451, 130, 132, 141, 221, 227, 263, 438, 456, 479, 503, 513]
del_list.sort(reverse = True)
for i in range(len(del_list)):
    del data[del_list[i]]
print(len(data))
df_document_list = pd.DataFrame(data,columns=['document'])
#df_document_list.to_csv (r'D:/myThesis/BIG data and LA paper/final_docs_without_outliers.csv', index = True, header=True)
