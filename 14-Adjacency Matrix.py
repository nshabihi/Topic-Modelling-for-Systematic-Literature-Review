###### Export data for drawing adjacancy matrix based on topics

# cosine for documents
rejoined_docs = []
for item in data_lemmatized:
    words = item
    j = " ".join(words)
    rejoined_docs.append(j)

#tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(rejoined_docs)

doc_term_matrix = X.todense()
dtm_df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
#############
cosine_doc = (cosine_similarity(doc_term_matrix, doc_term_matrix))

doc1 = []
doc2 = []
value = []
d = 0
s = 0
for i in range(len(cosine_doc)):
    for j in range(len(cosine_doc)):
        if cosine_doc[i][j] > 0:
            doc1.append(i+1)
            doc2.append(j+1)
            if (cosine_doc[i][j] < 0.09):
                value.append(0)
            else:
                value.append(cosine_doc[i][j])
            d += 1
            s += cosine_doc[i][j]
 

           
            
print("s:   "+str(s/((len(cosine_doc)*len(cosine_doc)))))
n= len(cosine_doc)
m=n+1
df_doc = pd.DataFrame({str(m): doc1, str(n): doc2 , str(d):value})
#df_doc.to_csv (r'D:/Topic-Modelling-for-Systematic-Literature-Review-main/Analysis files/doc_doc_adj.csv', index = False, header=True)


df_topic_docs = df_dominant_topic['Dominant_Topic']
#df_topic_docs.to_csv (r'D:/Topic-Modelling-for-Systematic-Literature-Review-main/Analysis files/doc_topic_association.csv', index = False, header=True)



#####################
## Initialize adjacancy matrix

import networkx as nx
from matplotlib import pyplot, patches

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
            
            
###########
### draw simple adjacancy matrix
from scipy import io
#use doc_doc_adj.csv to create the following .mtx file
A = io.mmread("D:/s3rmt3m1.mtx")
G = nx.from_scipy_sparse_matrix(A)
draw_adjacency_matrix(G)



##########
## draw adjacancy matrix with topics
## use doc_topic_association.csv to create the following .txt file
import numpy as np
from collections import defaultdict
from io import StringIO

def assignmentArray_to_lists(assignment_array):
    by_attribute_value = defaultdict(list)
    for node_index, attribute_value in enumerate(assignment_array):
        by_attribute_value[attribute_value].append(node_index)
    return by_attribute_value.values()

## Load in array which maps node index to dorm number
## Convert this to a list of lists indicating dorm membership
dorm_assignment = np.genfromtxt("D:/Bubble.txt", dtype="u4")

dorm_lists = assignmentArray_to_lists(dorm_assignment)

#print(dorm_assignment)
#print(dorm_lists)
## Create a list of all nodes sorted by dorm, and plot
## adjacency matrix with this ordering
nodes_dorm_ordered = [node for dorm in dorm_lists for node in dorm]
draw_adjacency_matrix(G, nodes_dorm_ordered, [dorm_lists],["blue"])
