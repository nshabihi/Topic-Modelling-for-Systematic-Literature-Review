## Pre Processing: this code is based on: 5-bigrams and 3-grams.py & 6-Frequency and TFIDF.py

# To lower before context-based text edit
data = [d.lower() for d in data]

#combine phrases with "-" between their words
data = [re.sub("-", "", sent) for sent in data]
#check:
#print(data[30])

## Pre Processing:  context based text edit, normalization, bigrams, 3-grams
## IMPORTANT: to find the words/phrases to replace in this section run Bigrams/3-grams
def multiple_replace(dict, text):
    # Creating a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    
    # Editing based on dictionary
    d = 0
    for t in text:
        text[d] = regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], t) 
        d += 1
    return text

if (True):#__name__ == "__main__": 
    dict = {
        
        # controlling tipos or mistakes in "this" dataset + removing genaral stop phrases based on bigram analysis!
        ';':' ',
        "programing" : "programming",
        'behaviour' : 'behavior',
        'behavioral' : 'behavior',
        'behavioural' : 'behavior',
        'behaviourist' :'behaviourist_',
        'behaviourism': 'behaviourism_',
        "/" : " / ",
        ";" : " ; ",
        ":" : " : ",
        "abstract only" : "",
        'through the': "",
        'the other': "",
        'about the': "",
        'however the': "",
        'there are': "",
        'the study': "",
        'the paper': "", 
        'this work': "",
        'recent years': "", 
        'have been': "", 
        'has been': "", 
        'the field': "",
        'the need': "", 
        'this study': "", 
        'for the': "", 
        'with the': "",
        'this paper': "",
        
# Noramalizing education levels
        "early childhood" : " earlychildhood ",
        "early ages" : " earlychildhood ",
        "kindergarten" : " earlychildhood ",
        "kindergartener" : " earlychildhood ",
        "under 6 years old" : " earlychildhood ",
        "preschool" : " earlychildhood ",
        "pre school" : " earlychildhood ",
        "pre-school" : " earlychildhood ",
        'kid' : ' child ',
        "children" : " child",
        "primary-school" : " earlychildhood ",
        "primary school" : " earlychildhood ",
        "primary education" : " earlychildhood ",
        "primary grade" : " earlychildhood ",
        
        "first grade" : " elementaryeducation ",
        "second grade" : " elementaryeducation ",
        "2nd grade" : " elementaryeducation ",
        "2nd-grade" : " elementaryeducation ",
        "third grade" : " elementaryeducation ",
        "3rd grade" : " elementaryeducation ",
        "3rd-grade" : " elementaryeducation ",
        "4th grade" : " elementaryeducation ",
        "4th-grade" : " elementaryeducation ",
        "fourth grade" : " elementaryeducation ",
        "fourth_grade" : " elementaryeducation ",
        "5th grade" : " elementaryeducation ",
        "5th-grade" : " elementaryeducation ",
        "fifth grade" : " elementaryeducation ",
        "elementary classroom" : " elementaryeducation ",
        "elementary school" : " elementaryeducation ",
        "elementary grade" : " elementaryeducation ",
        
        "6th grade" :" middleschool ",
        "6th-grade" :" middleschool ",
        "sixth grade" : " middleschool ",
        "7th grade" :" middleschool ",
        "7th-grade" :" middleschool ",
        "seventh grade" :" middleschool ",
        "8th grade" :" middleschool ",
        "8th-grade" :" middleschool ",
        "eighth garde" :" middleschool ",
        "intermediate grades" :" middleschool ",
        "intermediate education" :" middleschool ",
        "intermediate school" :" middleschool ",
        "middle grade" :" middleschool ",
        "middle school" :" middleschool ",
        "middle schools":" middleschool ",
        
        "undergraduate" : " higher_education ",
        "higher education" : " higher_education ",
        "highereducation" : " higher_education ",
        #" graduate " : " highereducation ",
        #" university" : " highereducation ",
        'universities' : 'university ',
        "college" : " higher_education ",
        "postsecondary" : " post_secondary ",#"postsecondary" : " highereducation ",
        "post secondary" : " post_secondary ",#"post secondary" : " highereducation ",
        "post-secondary" : " post_secondary ",#"post-secondary" : " highereducation ",
        

        "9th grade" : " highschool ",
        "9th-grade" : " highschool ",
        "ninth grade" : " highschool ",
        "10th garde" : " highschool ",
        "10th-garde" : " highschool ",
        "tenth garde" : " highschool ",
        "11th garde" : " highschool ",
        "11th-garde" : " highschool ",
        "eleventh garde" : " highschool ",
        "12th garde" : " highschool ",
        "12th-garde" : " highschool ",
        "twelfth garde" : " highschool ",
        "high school" : " highschool ",
        "high-school" : " highschool ",
        "high_school" : " highschool ",
        "secondary education" : " highschool ",
        "secondary school" : " highschool ",
        "secondary grade" : " highschool ",
        "k-12" : " ktwelve ",
        "k12" : " ktwelve ",
        
        
 #context based phrase edit (based on bigrams and 3-grams)             
        "project-based" : "project_based ",
        "project based" : "project_based ",                       
        "mathematical" : "math ",
        "mathematic" : "math ",        
        "critical thinking" : "criticalthinking",
        "algorithmic thinking" : " algorithmicthinking ",
        "problem-based learning" : "problembasedlearning",
        "evidence centred" : "evidence_centered",
        "text-based" : "text_based",
        "cognitive skill" : "cognitive cognitive_skill ",
        "problemsolving skill" : "problemsolvingskill ",
        "programming skill" : "programming programmingskill ",       
        "computer science" : "computer_science",
        "pupil" : "student",
        "learner" : "student",
        "learners" : "student",

        
        # Learning Based Algorithms
        'neural network' : "deep_learn  neural_network ",
        'neural net' : "deep_learn  neural_network ",
        'neural-network' : "deep_learn  neural_network ",
        "deep learning" : "deep_learn ",
        "deep learning model" : "deep_learn ",
        "deep_learning" : "deep_learn ",
        "deep-learning" : "deep_learn ",
        "deep neural network" : "deep_learn ",
        "deep artificial neural network" : "deep_learn ",
        "deepstealth" : " stealth deep_learn ",
        "deep network" : "deep_learn ",
        "machine learning" : "machine_learning",
        "machine-learning" : "machine_learning",
        "topic model" : "topic_modelling ",
        'topic detect' : "topic_modelling ",
        "algorithm" : "algorithm " ,        
        
        " cognition" : " cognitive",
        'academia' : 'academic',
        'academic success' : 'academic_success ',
        'academic performance' : 'academic_success ', 
        'academic achievement' : 'academic_success ',
        'student performance' : 'academic_success ',
        "robotics" : "robot ",
        "robitic" : "robot ",
        'automatic assessment' : 'automatic automatic_assessment ',
        'automatically' : 'automatic ',

        'computer scientists' : 'computerscientist ',
        'computational modeling' : 'computationalmodeling ',
        'computational model' : 'computationalmodeling ',
        'software engineering' : 'softwareengineering ',
        'analysis tool' : 'assessmenttool ',
        'testing tool' : 'assessmenttool ',
        'assessment tool' : 'assessmenttool ',
        'assessment instruments' : 'assessmenttool ',
        'assessment instrument' : 'assessmenttool ',
        
        'code analy' : 'codeanalysis ',
        
        'teacher development' : 'teacher_development',
        'teacher professional development' : 'teacher_development',
        'teacher training' : 'teacher_development',
        'teacher professional' : 'teacher_development',
        'teacher education' : 'teacher_development',
        "teachers' knowledge" : 'teacher_development',
        "teacher's knowledge" : 'teacher_development',
        'teaching-learning' : 'teacher_development',
        'educator': 'teacher ',
        'teachers': 'teacher',
        'class ' : 'classroom ',
        'classroom' : 'classroom ',
        
        'educational technology' : 'educational_technology ',
        'educational_technologies' : 'educational_technology ',
        
        'psychological' : 'psychology',
        'psychometric' : 'psychology',
        
        'statistically' : 'statistic',
        'statistical' : 'statistic',
        'steam' : 'stem',        
        'pedagogically' : 'pedagogy',
        'pedagogical' : 'pedagogy',
        
        'qualitatively' : 'qualitative',
        'eye tracking' : 'eye_tracking ',
        'gaze' : 'eye_tracking ',
        'log based' : 'log',
        'log-based' : 'log',
        'logs':'log ',
        'meta cognitive' : 'cognitive metacognitive',
        'meta-cognitive' : 'cognitive metacognitive',
        'debugging' : 'debug',
        'unplug' : 'unplugged',
        
        
        "mobile application" : " mobile mobileapp ",
        "mobile app" : " mobile mobileapp ",
        "smart phone" : "mobile ",
        "android device " : "mobile ",  
        
        "learning otucome" : "learning_otucome ",
        "artificial intelligence" : "artificial_intelligence",

        'website' : 'web',
        'gamified' : ' game ',
        'gamification ' : ' game ',
        'gaming': ' game ',
        'game': ' game ',
        "active learning" : " active_learning ",
        " tracing" : " trace ",
        " traces" : " trace ",
        "natural language processing" : " nlp ",
        "story" : " story ",
        "computer aided instruction" : "computer_aided_instruction",
        "learning pattern" : "learning_pattern pattern ",
        
    
        #'learning environment': 'learning_environment ',  #
        'collaborative learning': ' collaborative_learning collaborative ' ,
        'collaboration' : 'collaborative ',
       
        #'educational administrative': "educationaladministrative",  #
        'administrator' : 'administirative ',
        'electronic learning': "elearning",  #
        #'educational course': "educationalcourse ",  #
        'internet of things': " iot ",  #
        'internet-of-things': " iot ",  #
        'massive open online course' : 'mooc ',
        'mooc' : 'mooc ',

        'atrisk': "at_risk ",
        "at risk":"at_risk ",
        "at-risk":"at_risk ",
        'naïve bayes' : 'naive_bayes',
        'naive bayes' : 'naive_bayes',
        'naÃ¯ve bayes' : 'naive_bayes',
        
        'cloud computing': "cloud cloud_computing ",
        'cloud': ' cloud',
        'data model': 'data_model ', #
        'data governance' : 'data_governance governance',
        #'student learning': "studentlearning", #
        'business intelligence': 'business_intelligence ',
        'business intelligent': 'business_intelligence ',
        'data processing': 'data_processing ',
        'decision making': 'decision_making ',
        'decision-making': 'decision_making ',
        'decision maker': 'decision_making ',
        'decision-maker': 'decision_making ',
        'online learning': "online online_learning ",
        #'online courses': "online online_learning",
        'online learners':'online online_learning ',
        #'online_course':'online online_learning',
        'educational institutions': "educational_institutions institutions ",
        
        
        "educational data mining" : " edm ",
        #'data mining tool' : 'dataminingtool ',
        'software architecture' : 'software_architecture ', #
        'feature selection' : 'feature_selection ',
        'data visualisation' : 'visualisation data_visualisation ',#
        'open data': 'open_data ',
        'learning process':'learning_process ',
        'apache spark':'apache_spark ',
        'parallel processing':'parallel_processing ',
        'parallel comput':'parallel_processing ',
        
        'social network':'social_network ',
        'social media':'social_network ',
        'social-network':'social_network ',
        'social networking':'social_network ',
        'social-networking':'social_network ',
        #'collaborative filtering':'collaborativefiltering',
        'learning outcome':'learning_outcome ',
        'early warning':'early_warning ',
        
        #data
        'data warehouse':'data_warehouse ',# s
        'datawarehouse':'data_warehouse ',
        "data mining" : "data_mining",
        'data-mining' : "data_mining",
        
        #prediction
        "predictive model":"prediction predictive_model ",
        'predictive analytic' : 'predictive_analytic ',
        "performance prediciton":"prediciton performance_prediciton ",
        "grade prediction": "prediction grade_prediction ",
        "activity prediction": "prediction activity_prediction ",
        "drop out" : "drop_out ",
        "drop-out" : "drop_out ",
        'prediction algorithm':'prediction prediction_algorithm ',


        'information system':'information_system ', 
        'data privacy':'data_privacy privacy ',
        'privacy' : "privacy ",
        'recommender system':'recommender_system ',  
        'adaptive learning':'adaptive_learning', 
        'case study':'case_study',
        'case-study':'case_study',
        'case studies':'case_study',
        'case-studies':'case_study',
        
        'data visualization':'data_visualization ', 
        'data set':'data_set ', 
        'logistic regression':'logistic_regression ', 
        'academic analytic':'academic_analytic ', 
        'digital learning':'digital_learning ', 
        'learning activities':'learning_activity ',
        'learning activity':'learning_activity ',
        'log data':'log_data', 
         
        'supervised learning':'supervised_learning',
        'supervised classi' : 'supervised_learning classi',
        'supervised machi' : 'supervised_learning machi',
        'supervised data': 'supervised_learning data',
        'supervised cluster': 'supervised_learning cluster',
        
        'mobile learning':'mobile mobile_learning', 
        'technology enhanced learning':'technology_enhanced_learning', 
        'decision trees':'decision_tree ', 
        'decision tree':'decision_tree ', 
        'distance learning':'distance_learning', 
        'classification algorithm':'classification classification_algorithm ', 
        'virtual learning':'virtual_learning', 
        'feature extraction':'feature_extraction ', 
        'analytical model':'analytical_model ',
        
        #management:
        'learning management': 'learning_management ',
        'data management':'data_management ',
        'knowledge management':'knowledge_management ',
        'management system':'management_system ',
        
        #big data      
        "big data competitive":"big_data_competitive ",
        "big data application":"big_data_application ",
        "big data architecture":"big_data_architecture ",
        "big data tool":"big_data_tool ",
        "big data framework":"big_data_framework ",
        "big data technology":"big_data_technology ",
        "big data technologies":"big_data_technology ",
        "big-data competitive":"big_data_competitive ",
        "big-data application":"big_data_application ",
        "big-data architecture":"big_data_architecture ",
        "big-data tool":"big_data_tool ",
        "big-data framework":"big_data_framework ",
        "big-data technology":"big_data_technology ",
        "big-data technologies":"big_data_technology ",
        'massive data' : "big data ",
        'large data': "big data ",
        "bigdata" : "big data ",
        "big-data":"big data ",
        
        #learning analytics
        'learning analytic':'learning analytic ',
        'multimodal learning analytic':'multimodal multimodal_learning_analytic ',
        'learning analytic dashboard':'learning_analytic_dashboard dashboard ',
        'learning analytic integration':'learning_analytic_integration integration ',
        'ebook based learning analytic':'ebookbased ebook_based_learning_analytic ',
        'ebook-based learning analytic':'ebookbased ebook_based_learning_analytic ',#????????
        'integrating learning analytic':'learning_analytic_integration integration ',
        'learning analytic platform':'learning_analytic_platform ',  
        "data analytic" : "learning analytic ",
        'data analysis' : 'data_analysis ',
        "mixed method" : "mixed_method ",
        "mixed-method" : "mixed_method ",
        "map-reduce": ' map_reduce ',
        "map reduce": ' map_reduce ',
        
        #adaptive
        'personalization' : 'personalize ',
        'personalized' : 'personalize ',
        'personalizing' : 'personalize ',
                
        #special words
        'course management system' : ' cms ',
        'content management system' : ' cms ',
        'support vector machine': 'svm ',
        'svm':' svm ',
        "long short-term memory": ' lstm ',
        'lstm': ' lstm ',
        
        'data stream' : 'data_stream ',
        'click stream' : 'click_stream ',
        
        'self-regulation': 'self_regulated ',
        'self-regulated': 'self_regulated ',
        'mmla': ' mmla ',
        'multi modal': ' mmla ',
        'multi_modal' : ' mmla ',
        'multimodal': ' mmla ',
        
        'realtime': ' realtime ',
        'real time': ' realtime ',
        'real-time': ' realtime ',
        
        'statistics' : 'statistic',
        'patterns': 'pattern',
        'schools': 'school',
        'courses': 'course',
        'statistics' : 'statistic',
        'institutions': 'institution',
        'ethics': 'ethic',
        'ethical' : 'ethic',
        'interactions':'interaction',
        'curricular': ' curriculum ',
        'curricula': ' curriculum ',
        'librarian' : 'librarian library ',
        'libraries':'library',
        'librarianship' : 'librarian',

        'security' : 'security ',
        'systems' : 'system',
        'forums': 'forum',
        'webs':'web',
        'website': 'web ',
        'digitalization' : 'digitization',
        'multicultural': 'multicultural culture ',

        
        'optimisation' : 'optimisation ',
        'optimization' : 'optimisation ',
        'optimizing':  'optimisation ',
        'optimize'  : 'optimisation ',
        
        'classifier' : 'classification ', 
        'classified' : 'classification ',
        'classification' : 'classification ',
        'classify' : 'classification ',
        
        'computing': 'computation',
        
        'irregularity' : ' nonregular ',
        'regularity' : ' regular ',
        
        'epistemological ' : 'epistemic ',
        'epistemology ' : 'epistemic ',
        'epistemologies' : 'epistemic ',
        
        'summatively': 'summative',
        
        'gradient boost': 'gradient_boost ',
        'peer-to-peer' : 'peer_to_peer  peer ',
        'peer-review': 'peer_review peer ',
        'peer review': 'peer_review peer ',
        'data flow' : 'data_flow', 
        'data-flow' : 'data_flow',
        
        'self evaluation' : 'self_assessment ',
        'self-evaluation' : 'self_assessment ',
        'self assessment' : 'self_assessment ',
        'self-assessment' : 'self_assessment ',
        
        'matrix factorization' : 'matrix_factorization',
        'matrix decomposition' : 'matrix_decomposition',
        
        'video-based' : 'video video_based',
        'visual tracking' : 'visual_tracking tracking '
    }
#Replace (two times)    
data = (multiple_replace(dict, data))
data = (multiple_replace(dict, data))

# To lower after context-based text edit
data = [d.lower() for d in data]

#to combine phrases with "-" between their words
data = [re.sub("-", "", sent) for sent in data]
#check
#print(data[30])

#####################################################
### Stop word/ Bigram / 3-garm / Lematization
##sentence to word

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False, min_len=3, max_len= 40))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
#print(data_words[30])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words] for doc in texts]
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'PROPN','SYM']):
#def lemmatization(texts, allowed_postags=['PROPN']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

####stop word elimination and lemmatization
data_words_nostops = remove_stopwords(data_words)

### Form Bigrams
#data_words_bigrams = make_bigrams(data_words_nostops)

###Run these: 
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
#pip3 install spacy
#python3 -m spacy download en_core_web_sm
### And then run these in a python console.
#nlp = spacy.load("en_core_web_sm")
#doc = nlp("Text here")
nlp = spacy.load('en', disable=['parser', 'ner'])

#### Do lemmatization keeping only specific POSs
#data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN','SYM'])
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB','PROPN','SYM'])
#data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['PROPN'])

###check
#print(data_words_nostops[30])
#print(data_lemmatized[30])

