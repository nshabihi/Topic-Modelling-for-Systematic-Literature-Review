### Remove Duplicate: for some documents the titles are not "exactly" same,
#but still there are related to the same study.
#In this code, We want to find these type of ducuments.
#We chose 0.65 for the threshold to delete one of the each of two documents that are similar more than 0.65

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

data_M = X.todense()
data_df = pd.DataFrame(data_M, columns=vectorizer.get_feature_names())

# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine = (cosine_similarity(data_df, data_df))
#check number of features
#print(len(data_df.T))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

# add one of each two dcocuments with similarity of more than 0.65 to the delete list
dupli = []
del_list = []
for i in range(len(cosine)):
    for j in range(i+1,len(cosine)):
        if (cosine[i,j]) > 0.65:
            #uncomment to check each of the two duplicated documents
            #dupli.append(i)
            #dupli.append(str(i)+" - "+str(j))
            if len(data[i]) > len(data[j]):
                if j not in del_list:
                    del_list.append(j)
                else:
                    del_list.append(i)
            elif i not in del_list:
                del_list.append(i)
            else:
                del_list.append(j)
                
#print(dupli)
del_list.sort(reverse = True)

# delete dublicates based on cosine similarity
for i in range(len(del_list)):
    del data[del_list[i]]    
print("Number of remaining documents: " + str(len(data)))
df_document_list = pd.DataFrame(data,columns=['document'])
#df_document_list.to_csv (r'D:/MYT/BIG data and LA paper/without_outlier.csv', index = True, header=True)

### Run the following code to find a threshold for duplicate selection
##Selecting a proper cosine threshold for duplicate removal:
##Select Where there is "MIN change" in the number of found duplicates
#dupli = []
#del_list = []
#y=[]
#for c in range(40):
#    for i in range(len(cosine)):
#        for j in range(i+1,len(cosine)):
#            if (cosine[i,j]) > c*0.025:
#                if len(data[i]) > len(data[j]):
#                    if j not in del_list:
#                        del_list.append(j)
#                    else:
#                        del_list.append(i)
#                elif i not in del_list:
#                    del_list.append(i)
#                else:
#                    del_list.append(j)
#    y.append(len(list(set(del_list))))
#    #print(del_list)
#    del_list = []
#                
#
#import matplotlib.pyplot as plt 
#import numpy as np 
#x = []
#for j in range(40):
#    x.append(j*0.025)
#plt.xlabel('Cosine Treshold')
#plt.ylabel('Number of duplicates found')
#plt.plot(x, y)  
#plt.show() 
