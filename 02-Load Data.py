###  Load Data from bibtex and csv

from pybtex.database.input import bibtex
import pandas as pd
import re

bib_files = ["D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/acm.bib",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/IEEE-1.bib",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/IEEE-2.bib",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/ScienceDirect.bib",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/scopus.csv",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/wos.csv",
             "D:/Topic-Modelling-for-Systematic-Literature-Review-main/dataset/eric.bib"
            ]

title = []
abstract = []
keywords = []

for f in bib_files:
    if(f[-4:] == ".bib"):
        #open a bibtex file
        parser = bibtex.Parser()
        bibdata = parser.parse_file(f)
        for bib_id in bibdata.entries:
            b = bibdata.entries[bib_id].fields
            title.append(b["title"])
            abstract.append(b["abstract"])
            if "keywords" in b.keys():
                keywords.append(b["keywords"])
            else:
                keywords.append("")
                
    elif (f[-4:]== ".csv"):  #for one scopus csv
        if ("scopus" in f):
            df_Scopus = pd.read_csv(f)
            df_Scopus = df_Scopus[["Title", "Abstract", "Author Keywords", "Index Keywords"]]
            df_Scopus["Index Keywords"] = df_Scopus["Index Keywords"].fillna("")
            df_Scopus["Author Keywords"] = df_Scopus["Author Keywords"].fillna("")
            df_Scopus['Keywords'] = df_Scopus[['Author Keywords', 'Index Keywords']].apply(lambda x: ' ; '.join(x), axis=1)
            del df_Scopus["Author Keywords"]
            del df_Scopus["Index Keywords"]
            
        elif ("wos" in f):  #for one WoS csv
            df_WoS = pd.read_csv(f)
            df_WoS = df_WoS[["Article Title" ,"Abstract", "Author Keywords" , "Keywords Plus"]]
            df_WoS["Author Keywords"] = df_WoS["Author Keywords"].fillna("")
            df_WoS["Keywords Plus"] = df_WoS["Keywords Plus"].fillna("")
            df_WoS = df_WoS.rename(columns={"Article Title": "Title"})
            df_WoS['Keywords'] = df_WoS[['Author Keywords', 'Keywords Plus']].apply(lambda x: ' ; '.join(x), axis=1)
            del df_WoS["Author Keywords"]
            del df_WoS["Keywords Plus"]

df = pd.DataFrame({"Title": title, "Abstract": abstract, "Keywords":keywords})
#print(b.keys())
df = df.append(df_WoS, ignore_index=True)
df = df.append(df_Scopus, ignore_index=True)
#remove duplicates where titles are identical
df = df.drop_duplicates(subset="Title")
df = df.reset_index(drop=True)


#df.to_csv (r'D:/CTA/docs/docs.csv', index = True, header=True)


df['all'] = df[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' ; '.join(x), axis=1)
del df["Title"]
del df["Abstract"]
del df["Keywords"]

data = df["all"].tolist()
print(data[0])
# Remove Emails
data = [re.sub('\S*@\S*\s?', ' ', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", " ", sent) for sent in data]
data = [re.sub("\n", " ", sent) for sent in data]
data = [re.sub("œ", " ", sent) for sent in data]
data = [re.sub("â", " ", sent) for sent in data]
data = [re.sub("\200", " ", sent) for sent in data]
data = [re.sub("\235", " ", sent) for sent in data]
data = [re.sub("\231s", " ", sent) for sent in data]

data = [d.lower() for d in data]
print(len(data))
