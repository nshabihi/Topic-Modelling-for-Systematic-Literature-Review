#Delete documents with less than 4 keywords
n = 0
doc_del_list = []
for i in range(len(data_lemmatized)):
    if len(data_lemmatized[i]) < 4:
        print(str(i)+ " :   "+ str(len(data_lemmatized[i])))
        print(data_lemmatized[i])
        doc_del_list.append(i)
        n += 1
        
doc_del_list.sort(reverse=True)

temp = data_lemmatized
for i in range(len(doc_del_list)):
    temp.remove(temp[doc_del_list[i]])
