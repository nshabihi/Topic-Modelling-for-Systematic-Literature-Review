###### Export data for network analysis
## make network nodes csv ##

# get the list of top words and their scores in the topics
topic_num = 6
word_list_unique = []
word_score_dict = {}
word_topic_dict = {}
for i in range(topic_num):
    
    #temp_topic = optimal_model.print_topics(num_words=5)[i][1]
    temp_topic = lda_model.print_topics()[i][1]
    temp_list = temp_topic.split(" + ")
    temp_list = [x[:-1] for x in temp_list]
    for t in temp_list:
        w = t.split('*"')
        if str(w[1]) not in word_list_unique:
            word_score_dict [str(w[1])]=[float(w[0])]
            word_topic_dict [str(w[1])]=[i+1]
            word_list_unique.append(str(w[1]))
        elif word_topic_dict [str(w[1])] < [float(w[0])]:
            word_score_dict [str(w[1])]=[float(w[0])]
            word_topic_dict [str(w[1])]=[i+1]

word_topic = []
word_score = []
for word in word_list_unique:
    word_topic.append(word_topic_dict[word][0])
    word_score.append(word_score_dict[word][0]*2000)

df_nodes = pd.DataFrame({"name": word_list_unique, "group": word_topic , "nodesize":word_score})
df_nodes.to_csv (r'D:/MYT/BIG data and LA paper/Analysis files/nodes.csv', index = False, header=True)

## make source destination csv ##
### cosine for words
tdm_df = dtm_df.T
print(tdm_df.shape) 

#delete column and row from tdm matrix if a word are not in a top word in a topic
for ind in tdm_df.index:
    if ind not in word_list_unique:
        tdm_df = tdm_df.drop(ind)

# keep the list of words based on tdm matrix indexes
tdm_word_list = []
for ind in tdm_df.index:
    tdm_word_list.append(ind)

#similarity between words
cosine_tdm = (cosine_similarity(tdm_df, tdm_df))

#make source destination csv
w1 = []
w2 = []
sim = []
words = []
source = []
target = []
d = 0
for i in range(len(tdm_word_list)):
    for j in range((i+1),len(tdm_word_list)):
        if cosine_tdm[i][j] > 0.15 and i != j:
            sim.append((cosine_tdm[i][j])*200)
            source.append(tdm_word_list[i])
            target.append(tdm_word_list[j])

df_source_target = pd.DataFrame({"source": source, "target": target , "value":sim})
df_source_target.to_csv (r'D:/MYT/BIG data and LA paper/Analysis files/df_source_target.csv', index = False, header=True)
