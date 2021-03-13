######################################################
###Removing unrelevant words for FEATURE SELECTION
### method (the following methods are used to extract the list of unrelevant words):
## 1.check frequency, 
## 2.tfidf, 
## 3.select no more than 15 words from each document
## Check POSs one by one
## Select different number of words from each ducoments and check the words based on their frequency and tfidf

## based on the above analysis/methods the follwing code has been written!


######### word removal function
def remove_words(texts, del_list):
    texts = [[word for word in doc if word not in del_list] for doc in texts] 
    return texts

################### low frequent removal based on TFIDF
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
del_list_tfidf = []
for i in range(len(ranked_terms_tfidf)):
    if ranked_terms_tfidf.iloc[i,1]<0.25:
        del_list_tfidf.append(ranked_terms_tfidf.iloc[i,0])

print("The word number before word_removal "+ str(len(ranked_terms_tfidf)))
print("the word number to be deleted based on TFIDF: " + str(len(del_list_tfidf)))
data_lemmatized = remove_words(data_lemmatized, del_list_tfidf)

################### low frequent removal based on TFIDF
rejoined_docs = []
for item in data_lemmatized:
    words = item
    j = " ".join(words)
    rejoined_docs.append(j)
    
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
del_list_count = []
for i in range(len(ranked_terms_freq)):
    if ranked_terms_freq.iloc[i]['rank'] < 2:
        del_list_count.append(ranked_terms_freq.iloc[i]['term'])

print("the word number to be deleted based on COUNT: " + str(len(del_list_count)))
data_lemmatized = remove_words(data_lemmatized, del_list_count)

###################delete unuseful adjectives (found based on pos="ADJ" in lemmatizer)       
del_list_adj=['hard', 'compelling', 'embed', 'evident', 'bibliometric', 'contingent', 'distinct','thorough','distributive', 'generate', 'nanomaterial', 'happy',
 'connected', 'congestive', 'dramatic', 'moderate', 'structuralfunctional', 'focal', 'former', 'finegrained', 'varied', 'casual', 'fifth', 'material',
 'visible', 'false', 'facilitate', 'boundary', 'exemplar', 'black', 'wellknown', 'wellreceive', 'western', 'biometric', 'extracting', 'express', 
 'willing', 'wireless', 'workflowbase', 'valid', 'utilize', 'chinese', 'traceable', 'exhaustive', 'transformative', 
 'unable', 'uncertain', 'undermine', 'underrepresented', 'civic', 'unlocking', 'mechanic', 'fit', 'unsuccessful', 'measurable', 'sociomaterial', 
 'convenient', 'neoliberal', 'societal', 'paral', 'intangible', 'instructionintensive', 'enhanced', 'instant', 'demographical','outline',
 'informationtheoretic','prospective', 'protect','environmental', 'quick', 'rational', 'alternate', 'integral', 'abstract', 'exponential',  'plural',
 'persistent', 'intermediate', 'investigate', 'differentiate', 'endless', 'perceive', 'inter', 'didactic', 'prevalent','precommercial', 
 'developmental', 'patient','developed', 'past', 'onprocessordie', 'dark', 'realistic', 'sharable', 'immutable', 'costeffective', 'senegal', 
 'nonnegative', 'immediate', 'noninstructional', 'nonexistent', 'identifiable', 'correlative', 'correlate', 'secondary','neutral', 'humanistic', 
 'legislative', 'architectural', 'singular', 'slow', 'pet', 'contradictory', 'holistic', 'normal','apparent', 'requirement', 'affective', 
 'layered', 'indirect', 'discursive', 'incorporate', 'remedial', 'inconsistent', 'noticeable', 'discussionintensive', 'disparate', 
 'rewarding', 'curious', 'cumulative', 'rigorous', 'incoming', 'round', 'cultural', 'scholarly', 'malicious','latter', 'mainstream', 
 'augment', 'bigquery', 'frequent', 'forthcoming', 'synchronous', 'fitting', 'micro', 'heis', 'harmful', 
 'intensive', 'incomplete', 'familiar', 'concentrated', 'static', 'concrete',
 'spdi', 'expensive', 'numerical', 'contexts', 'contextual', 'continual', 'old', 'equitable', 'equal', 'correct', 'optimal', 'indispensable',
 'biomedical', 'verifiable', 'ready', 'thematic', 'derive', 'detect', 'adaptable', 'viable', 'ambient', 'total', 'senior', 'alternative', 'select',
 'prerequisite', 'local', 'exciting', 'imperative', 'mean', 'enable', 'fuzzy', 'english', 'foreign', 'mutual', 'national','explosive',
 'evaluative', 'everyday', 'discriminant', 'probabilistic', 'publish', 'influential', 'internal', 'lean', 'kind', 'proofofconcept',
 'prescriptive', 'informational','actual', 'additional', 'prominent', 'weekly', 'twin','collect', 'commercial', 'preliminary', 'difficult',
 'ubiquitous', 'capable', 'next', 'strong', 'typical', 'temporal', 'promising', 'operational','entire', 'close', 'explicit', 'central', 'interested',
 'considerable', 'fast', 'feasible', 'simple', 'concerned', 'productive', 'spatial', 'polytechnic', 'authentic', 'easy', 'functional', 'relative',
 'widespread', 'eager', 'least', 'legal', 'disruptive', 'essential', 'unprecedented', 'perspective', 'proper', 'geographical', 'detailed', 'fine',
 'generic', 'serendipitous', 'daily', 'obvious', 'poor', 'fundamental', 'constitutive', 'third','associate','true', 'tremendous', 
 'beneficial', 'enough', 'clinical', 'extensive', 'inform', 'elaborate', 'compute', 'probable', 'prior', 'lead', 'prompt','applicable',
 'thestudent', 'brief', 'remarkable', 'consistent', 'constant', 'secure', 'defective', 'direct', 'reveal', 'respective', 'principal', 'late', 
 'inclusive', 'external', 'financial', 'original', 'focus', 'logistic', 'participant', 'implicit', 'negative', 'hot', 'likely', 'reliable',
 'robust', 'socioeconomic','annual', 'urgent', 'unintended', 'longitudinal', 'unified',  'psycholinguistic',
 'compound','real', 'previous', 'free', 'scientific', 'full', 'global', 'suitable','complete', 'timely',
 'advanced', 'vast', 'right','overall', 'historical', 'low', 'appropriate', 'formal', 'standard', 'single', 'second', 'certain', 'unique', 'rapid',
 'broad', 'whole', 'basic', 'special', 'related', 'powerful', 'initial', 'interesting', 'popular', 'industrial', 'synthetic', 'long', 'crucial',
 'informed', 'helpful', 'similar', 'little', 'module', 'accurate', 'comparative', 'successful', 'positive', 'serious', 'top', 'private', 'rich',
 'accessible', 'natural', 'modern', 'novel', 'well', 'emotional', 'conventional', 'enormous', 'profile', 'hybrid', 'flexible', 'soft', 
 'methodological', 'metric', 'economic', 'sustainable', 'numerous', 'demographic', 'sophisticated', 'contemporary', 'sufficient',
 'substantial', 'clear', 'causal', 'adequate', 'random', 'average', 'primary', 'vital','massive', 'professional', 'variable', 'common', 'semantic', 
 'conceptual', 'much', 'due', 'general', 'small', 'meaningful', 'last', 'characteristic', 'wide', 'first', 'medical', 'comprehensive', 'necessary',
 'systematic', 'great', 'medium', 'major', 'heterogeneous','human', 'efficient', 'good', 'early', 'huge', 'able', 'critical', 'particular',
 'significant', 'main', 'relevant', 'complex', 'big', 'available', 'several', 'recent', 'possible', 'specific','new', 'different', 'present',
 'potential', 'large', 'current', 'various', 'many', 'important', 'future', 'high', 'key', 'effective', 'useful','educational']
print("the word number for ADJ removal: " + str(len(del_list_adj)))
data_lemmatized = remove_words(data_lemmatized, del_list_adj)


###################delete unuseful adjectives (found based on pos="VERB" in lemmatizer)       
del_list_verb=['utilise', 'submit', 'comprise', 'welcome', 'function', 'leave', 'concept', 'lm', 'divide', 'flip', 'accept', 'dominate','assign', 'resource', 
 'literature', 'compose', 'survey', 'induce', 'calculate', 'feel', 'improvement', 'invest', 'allocate', 'example', 'place', 'succeed', 'enter',
 'satisfaction', 'outperform', 'tell', 'imagine', 'remove', 'hear', 'replace', 'join', 'issue', 'data', 'name', 'fill', 'archive', 'collate', 
 'object', 'disclose', 'notice', 'outsource', 'impede', 'grasp', 'randomize', 'mature', 'rfid', 'internet', 'locate', 'impose', 'coordinate', 
 'pass', 'entail', 'initiate', 'overcome', 'correspond', 'stage', 'underpin', 'own', 'selfpace', 'wrangle', 'content', 'afford', 'depict', 'slide', 
 'capabilitie', 'push', 'query', 'accelerate', 'difference', 'devise', 'prevent', 'minimize', 'encounter', 'infer', 'pave', 'accompany', 'constrain',
 'stack', 'neglect', 'rhizome', 'specialize', 'differ', 'debate', 'author', 'empower', 'fiveside', 'scatter', 'migrate', 'resolve', 'unveil', 'foster',
 'overlap', 'happen', 'timestampe', 'layer', 'communicate', 'complement', 'charge', 'figure', 'exchange', 'ground', 'render', 'constitute', 'fall',
 'search', 'courseware', 'pull', 'processoriente', 'launch', 'struggle', 'comment', 'implementation', 'clarify', 'adjust', 'reason', 'formalize', 
 'predefine', 'navigate', 'count', 'domain', 'studentcentere','imply', 'theorybase', 'contrast', 'fund', 'assert', 'nuance','bridge', 'consume',
 'agree', 'categorise', 'nee', 'acknowledge', 'pursue', 'bibliographie', 'conference', 'reward', 'justify', 'pre', 'invite', 'encompass', 'illdefine',
 'theorize', 'mitigate', 'focusse', 'master', 'prevail', 'acm', 'realitysupporte', 'prohibit', 'performance', 'wish', 'craft', 'ten', 'hinder',
 'retrieve', 'careeroriente', 'conceptualize', 'inference', 'devote', 'further', 'objective', 'overlook', 'exceed', 'part', 'realise', 'intertwine',
 'eventcentre', 'finegraine', 'envision', 'transcend', 'rate', 'witness', 'prefer', 'acquire', 'universidad', 'tout', 'concentrate', 'designer', 
 'complexitie', 'respond', 'effect', 'suppose', 'endorse', 'list', 'complicate', 'systematicliterature','possibilitie', 'dematelbase', 'deficiencie',
 'post', 'instantiate', 'consequence', 'sample', 'permit','evidence', 'stand', 'assume', 'hope', 'core', 'compromise', 'talk', 'care',
 'interrelationship', 'lose', 'satisfy', 'originate', 'bolster', 'return', 'ecosystem', 'stimulate', 'user', 'supplement', 'internetbase', 'survive',
 'humanle', 'widerange', 'burgeon', 'eliminate', 'wikis', 'commence', 'incentive', 'centralize','usergenerate', 'favor', 'redesign', 'unlock',
 'sensemake', 'everincrease', 'avoid', 'appreciate', 'invent', 'hei', 'democratize', 'mlr', 'advocate', 'claim', 'consolidate', 'learnerrelate',
 'trace', 'parse', 'nudge', 'lowperforme', 'fear', 'undergo', 'phase', 'harness', 'customise', 'showcase', 'note', 'couple', 'panelist', 'preserve',
 'learningoriente', 'accomplish', 'librarie', 'fellow', 'restructure', 'necessitate', 'competencybase', 'authenticate', 'networkbase', 'time',
 'maximize', 'mix', 'decrease', 'firstyear', 'signalize', 'diestacke', 'prefetche', 'suit', 'textmine', 'bootstrappe', 'unexplore', 'salient', 'effort', 'insight', 'convert', 'stay', 'machinegenerate', 'partition', 'trajectorie', 'competence', 'strive', 'filter',
 'workload', 'coin', 'shrink', 'keystroke', 'skip', 'coincide', 'type', 'tackle', 'belong', 'appeal', 'underperform', 'technologymediate', 
 'hypothesize', 'redefine', 'learnergenerate', 'studentface', 'power', 'grant', 'lor', 'stateoftheart', 'distil', 'prefetch', 'fuel', 'basis',
 'timedelaye','learn', 'use', 'provide', 'improve','make', 'analyze', 'develop', 'include', 'identify', 'propose', 'teach', 'explore', 'apply',
 'discuss', 'become', 'may', 'understand', 'help', 'increase', 'need', 'aim', 'create', 'find', 'show', 'relate', 'support', 'emerge', 'design',
 'set', 'describe', 'conduct', 'could', 'give', 'take', 'allow', 'require', 'consider', 'offer', 'process', 'examine', 'implement', 'obtain', 
 'achieve', 'introduce', 'extract', 'address', 'enhance', 'exist', 'result', 'suggest', 'compare', 'evaluate', 'integrate', 'grow', 'face', 'call', 
 'distribute', 'draw', 'reserve', 'produce', 'review', 'regard', 'analyse', 'demonstrate', 'accord', 'involve', 'challenge', 
 'build', 'feature', 'gather', 'blend', 'follow', 'hide', 'approach', 'promote', 'know', 'would', 'share', 'combine', 'seek', 'represent', 'perform',
 'bring', 'contribute', 'work', 'indicate', 'adopt', 'conclude', 'drive', 'come', 'assess', 'solve', 'reflect', 'must', 'look', 'employ', 'begin',
 'capture','advance', 'establish', 'report', 'study', 'argue', 'assist', 'change', 'view', 'define', 'carry', 'measure','see', 'affect', 'choose',
 'handle', 'test', 'discover', 'explain', 'meet', 'evolve', 'store', 'gain', 'remain', 'highlight', 'engage', 'manage', 'ensure', 'think', 'start',
 'serve', 'influence', 'benefit', 'record', 'raise', 'receive', 'target', 'illustrate', 'link', 'get', 'determine', 'interact',
 'do', 'write', 'extend', 'refer', 'concern', 'train', 'contain', 'expand', 'exploit', 'validate', 'rely', 'put', 'underlie', 'continue', 'lack',
 'adapt', 'order', 'form', 'promise', 'seem', 'consist', 'field', 'keep','occur', 'deal', 'research', 'depend', 'deliver', 'deploy', 'recognize',
 'move', 'run', 'experience', 'expect', 'intend', 'try','add', 'prepare', 'align', 'mention', 'purpose','can', 'range','rethink','question',
 'end', 'leverage', 'organize', 'operate', 'limit', 'drop', 'context', 'connect', 'appear', 'read', 'attract', 'miss', 'plan', 'spend', 'attempt',
 'cover', 'shape', 'control', 'cope', 'inspire', 'arise', 'detail', 'rise', 'believe','prove','surround', 'mediate', 'realize', 'modify', 'maintain',
 'orient','say', 'point','benchmarke','boost', 'cause', 'encourage', 'usage', 'embrace', 'demand', 'device', 'uncover', 'vary', 'attend', 'unify',
 'respect', 'waste','trend', 'hold', 'socalle', 'lie', 'tailor', 'answer', 'ask', 'intervene', 'simplify', 'number', 'amount','unfold',
 'save', 'spread', 'tool', 'verify', 'translate', 'attain', 'role', 'position', 'format','ignore', 'disrupt', 'confirm', 'conceive','pay',
 'impact', 'want', 'access','turn', 'construct', 'advise', 'go', 'tend']
print("the word number for VERB removal: " + str(len(del_list_verb)))
data_lemmatized = remove_words(data_lemmatized, del_list_verb)


###################delete unuseful NOUNs (found based on pos="NOUN" in lemmatizer)       
del_list_noun=['commitment', 'adherence', 'variation', 'spearman', 'workforce', 'vector', 'specialist', 'activation','viewpoint',
 'setup', 'commerce', 'advocacy', 'client', 'clock', 'advertising', 'classroomlike', 'silicon', 'signal', 'citizenship', 'circle', 'weather', 'chunk',
 'column', 'commentary', 'terabyte', 'universalism', 'characteristics', 'bit','avenue', 'birth', 'authority', 'threshold', 'tank', 'table', 
 'assistant', 'systematization', 'authorit', 'tip', 'attentiveness', 'attack', 'traction', 'backdrop', 'balance', 'biasvariance', 'bandwidth', 
 'template', 'batch', 'terminology', 'basket', 'bde', 'baseline','benchmark', 'bank', 'beneficiary', 'thatstudentinteraction', 'tradeoff', 'trail',
 'sport', 'anonymization', 'ubiquity', 'categorize', 'app', 'centre', 'antecedent', 'cfsfdphd', 'steer', 'utilisation', 'trainer', 'ankle', 
 'statement', 'alter', 'channel', 'stance', 'chapter', 'applying', 'catalog', 'card', 'appreciation', 'brand', 'substation', 'subpopulation', 
 'aspiration', 'broadband', 'articulation', 'campaign','array', 'candidate', 'approximation','capital','desire', 'conceptualization', 'gauteng',
 'ontask', 'generalpurpose', 'grain', 'guarantee', 'norm', 'niobium', 'niche', 'nextgeneration', 'nextbasket', 'haar', 'neighbour', 'necessity', 
 'multistage', 'heat', 'multicore', 'mse', 'higherorder', 'highquality', 'opening', 'futurelearn', 'parent', 'person', 'permission', 'percent',
 'penetration', 'facilitator', 'familiarity', 'patron', 'pathway', 'feed', 'food', 'furthermore', 'force', 'formalism', 'formulation','frontier',
 'origin', 'orientation', 'option', 'funding', 'hightech', 'month', 'history', 'linkage', 'maintenance', 'machining', 'iteration', 
 'judgment', 'loading', 'knowhow', 'labor', 'licence', 'labour', 'levy', 'leuven', 'letter', 'invitation', 'maneuver', 'hour', 'mobility',
 'incompatibility', 'metareview', 'incorporation', 'interpretability', 'maximise', 'inspiration', 'matriculation', 'instrumentation', 'integrating',
 'integrity', 'mark', 'interplay', 'excavation', 'critique', 'remedy', 'registration', 'deficiency', 'criticality', 'density', 'department',
 'reading', 'rationality', 'representationalism', 'credit', 'scenery','confluence', 'conjunction', 'consortium', 'scrutiny', 'contact', 'contention',
 'continuity', 'scanning', 'creator', 'scaling', 'copying', 'root', 'coverage', 'coword', 'race', 'eventstream', 'enrich', 'edition', 'premier', 
 'elaboration', 'poverty', 'potentiality', 'postcolonialism', 'energy', 'entity', 'pressure', 'plurality', 'equip', 'escience', 'plant', 'piloting', 
 'etraining', 'pillar', 'press', 'prevalence', 'dependency', 'proportion', 'learnification', 'pursuit', 'diffusion', 'proxy', 'director', 
 'discrimination', 'print', 'disruption', 'distillation', 'professionalism', 'door', 'procurement', 'drummer', 'duration','leaner', 'splitting', 
 'novelty','specialty', 'wellbeing', 'chance', 'toolbox', 'viewer', 'elitism', 'nextterm', 'email', 'processor', 'metaphor', 'marketing', 'minority',
 'mass', 'thousand', 'milestone', 'disaster',  'directive',  'match', 'meal', 'abundance', 'meeting', 'depth', 'advice', 'discusse', 'suitability',
 'mahout', 'ease', 'arena', 'driving', 'driver', 'stock', 'veracity', 'streaming', 'momentum', 'dissemination', 'display', 'variance', 'alert', 
 'manner', 'subset', 'aid', 'indication', 'radiography', 'contract', 'gathering', 'readiness', 'weakness', 'realization', 'wave', 'wealth', 'fusion',
 'canvas', 'fulfill', 'pocket', 'recovery', 'generating', 'beverage', 'poster', 'bibliography', 'humanity', 'hype', 'probe', 'profit', 'cop', 
 'bottleneck', 'heterogeneity', 'heart', 'gold', 'preview', 'promotion', 'identifie', 'imbalance', 'branch', 'contradiction', 'correctness', 
 'burden', 'leader', 'recruitment', 'intention', 'schedule', 'formation', 'perceptron','room', 'interval', 'trajectory', 'expose', 'persistence',
 'pasteurization', 'keyword', 'excellence', 'custom', 'everincreasing', 'customer', 'panel', 'rhetoric', 'intent', 'feeding', 'informatic', 
 'placement', 'transaction', 'consistency', 'conflict', 'pharmacovigilance', 'confidence', 'criticism', 'reproducibility', 'resolution', 
 'responsibility', 'host', 'reader', 'translation', 'transparency', 'hotspot', 'efficacy', 'agent', 'taxonomy', 'danger', 'correspondence', 
 'productivity', 'proliferation', 'granularity', 'horizon', 'server', 'aggregation', 'advising', 'protocol', 'differentiation', 'corporatization',
 'corelet', 'session', 'asset', 'friction', 'recommend', 'supply', 'assemblage', 'ecommerce', 'facetoface', 'row', 'feasibility', 'association', 
 'reliance', 'relevance', 'endeavor', 'allocation', 'fiction', 'execution', 'exclusion', 'dataveillance',  'alignment', 'footprint', 'fraction',
 'adopter', 'nation', 'cleaning', 'investment', 'population', 'characterization', 'literacy', 'moment', 'location', 'validation', 
 'uncertainty', 'accessibility', 'neighbor', 'utility', 'pitfall', 'practicality', 'picture', 'paradigms', 'ownership', 'competitiveness', 'owner',
 'update', 'chain', 'mastery','causality', 'length', 'journal','convergence', 'compliance', 'fashion', 'generator', 'modification','functionality', 
 'respondent', 'offering', 'attendee', 'reliability', 'one', 'region', 'retrieval', 'obstacle','scene', 'character', 'miner', 'priority', 'advent',
 'trading', 'equipment', 'equality','anomaly', 'manipulation', 'majority', 'medicine', 'worker', 'box', 'million', 'explanation', 'intersection',
 'constraint', 'existence','elicit', 'transcript', 'administration', 'distribution', 'summary','proceeding','estimate', 'belief', 'submission', 
 'presence', 'sheet', 'series', 'preparation', 'absence', 'agenda', 'condition', 'thank', 'line', 'specification', 'center', 'justice',
 'reporting', 'vendor', 'version', 'accreditation', 'date', 'extension', 'self', 'institutes', 'usability', 'significance', 'provider', 'man', 
 'cycle', 'emphasis', 'applicability', 'member', 'strength', 'ecology', 'matter', 'astudent', 'assumption', 'validity', 'mission', 'percentage',
 'disposition', 'personnel','publishing', 'accountability', 'definition', 'directory', 'progression', 'thread', 'connection', 'family', 'argument',
 'inclusion', 'hypothesis', 'assistance', 'assurance', 'conception', 'healthcare', 'estimation', 'barrier', 'flexibility', 'managementsystem', 
 'contextualization', 'frame', 'quantity', 'message', 'code', 'opendata', 'book', 'choice', 'speed', 'inquiry', 'usefulness', 
 'trial','utilization','error','philosophy', 'enterprise', 'originality', 'popularity', 'output', 'century', 'laboratory', 'economy','cost',
 'advancement', 'interview', 'unit', 'meaning', 'portion', 'week', 'preference','safety', 'dimension', 'developer', 'scholar', 'phenomenon', 
 'particle', 'criterion', 'rating', 'traffic', 'drug', 'attribute', 'section', 'ing', 'extraction', 'writing', 'involvement', 'building',
 'index', 'sharing', 'instance', 'status', 'expansion','expert', 'attrition', 'exploitation', 'interoperability', 'shift', 'datavisualisation',
 'vision', 'consent', 'overview', 'thinking', 'day', 'sense', 'woman', 'guidance', 'lot', 'provision', 'procedure', 'prospect', 'period', 'forest',
 'document', 'word', 'transform','citizen', 'completion', 'testing', 'extent', 'acceptance','audience', 'revolution', 'lab', 'landscape', 'guide',
 'transformation', 'reform','acquisition', 'surveillance', 'conclusion', 'complexity', 'response', 'conversation', 'representation', 'capacity', 
 'age', 'other', 'measurement', 'life', 'affordance', 'company', 'fact', 'product', 'planning', 'proposal', 'description', 'generation', 'tension',
 'transfer', 'production', 'attendance','foundation', 'scholarship', 'evolution', 'comparison', 'similarity', 'size', 'exploration','exam', 
 'background', 'health', 'image', 'identification', 'element', 'creation', 'delivery', 'file', 'subject', 'selection', 'advantage', 'theme',
 'perception', 'lecture', 'idea', 'paradigm', 'gap', 'hand', 'protection', 'rule', 'reality', 'emergence', 'operation', 'combination', 'introduction',
 'participation', 'category', 'actor', 'efficiency', 'engine', 'account', 'site', 'reference', 'publication', 'drawing', 'investigation', 'capability',
 'edx', 'pencil', 'input', 'availability', 'nature', 'event','campus', 'failure', 'market', 'growth', 'score', 'today', 'ability', 'progress', 
 'faculty', 'setting', 'adoption', 'structure', 'practitioner', 'scenario', 'correlation','competency', 'situation', 'infrastructure', 'addition', 
 'industry', 'degree', 'space', 'era', 'people', 'consideration', 'possibility', 'item', 'state', 'style', 'staff', 'initiative','variety', 
 'difficulty', 'mechanism', 'decade', 'limitation', 'direction','relation','article', 'area', 'finding', 'outcome', 'year', 'source', 'sector', 
 'case', 'discussion', 'goal', 'collection', 'relationship', 'attention','accuracy', 'understanding', 'aspect', 'effectiveness', 'paper', 'path', 
 'importance','development', 'level', 'edm', 'science']
print("the word number for noun removal: " + str(len(del_list_noun)))
data_lemmatized = remove_words(data_lemmatized, del_list_noun)


###################delete unuseful PROPN (found based on pos="PROPN" in lemmatizer)       
del_list_propn=['taylor', 'francis','ieee', 'millipede', 'legacy', 'lak', 'outcomes', 'south', 'inc', 'ltd', 'kingdom', 'google', 'identifies', 'scope', 
 'linemen', 'harnessing', 'michael', 'peters', 'value', 'licensee', 'andalusia', 'profuturo', 'rattle', 'wikipedia', 'hellenic', 'district',
 'harris', 'coldstart', 'held', 'resources', 'tigris', 'tianjin', 'theory', 'synthesize', 'teresa', 'york', 'xuetangx', 'plata', 'tendency',
 'vikor', 'udacity', 'scfhs', 'uml', 'visekriterijumsko', 'results', 'rangiranje', 'springer', 'tlcs', 'spark', 'tom', 'practical',
 'transversal', 'trondheim', 'twitter', 'proceedings', 'processlevel', 'dilemmas','deployment', 'depart', 'dematel', 'cube', 'crosscorelet', 
 'council', 'elsevier', 'eastern', 'east', 'ean', 'dyslexia', 'dyscalculia', 'compostela', 'api', 'bda', 'bayesnet', 'arrival', 'apriori', 'appraisal',
 'apache', 'andalusian', 'accreditations', 'beiesp', 'blueprint', 'commission', 'coala', 'city', 'citation', 'calculation', 'cache', 'bpm', 
 'fed', 'maximization', 'mauritius', 'mahara', 'liferay', 'leprove', 'latin', 'lasso', 'larc', 'kwazulunatal', 'mlaas', 'modal', 
 'nsagencyber', 'nrc', 'northern','neoliberalism', 'nearest', 'movement', 'moscow', 'moodie', 'keywords', 'ger', 'hollywood', 'hlm', 'hho', 'harvard',
 'gpa', 'globe', 'gdpr', 'jrip', 'gdelt', 'fouryear', 'fora', 'fms', 'humanmachine', 'iadis', 'idefa', 'igi', 'jeanluc', 'intelligentization',
 'indexes', 'kompromisno', 'board','nexus','systematic_literature','informa','rights','students','tel','ofstudent','modules','environment',
 'environments','structural','bias','solutions','deleuze','public','system','inclass','semester','technical','method','economics','way','login','los',
 'service','communications','bkt','expectations','pearson','applied','ondemand','slr','evaluation','art','features','individual','systematic',
 'literature','act', 'uclm', 'kratos', 'offline', 'stratum', 'palmer', 'parameter', 'relational', 'kmf', 'intelligence', 'enrollment', 'ebd', 
 'software', 'network', 'training', 'media', 'rre', 'machines', 'noncoldstart', 'semanticallyrich', 'closedloop', 'bidel', 'bmlas', 'mimd', 'renai',
 'videomark', 'methods', 'middle', 'aacsb', 'lrnn', 'hawks', 'cheat', 'sta', 'west', 'isd', 'contentaware', 'diy', 'mdvc', 'ksa', 'energyefficient',
 'ssmc', 'flowcontrol', 'booklooper', 'individual', 'ontological', 'inputlog', 'vacc', 'aware', 'datos', 'learnsphere', 'frequency', 'pedagogic',
 'critiques', 'career', 'mechanisms', 'unisa', 'longterm', 'oulad', 'lea', 'aol', 'weka', 'quantitative', 'oer', 'devices', 'gan', 'generative',
 'technologies', 'big_data_technology', 'genie', 'elo', 'cohen', 'lmooc', 'analyser', 'mdpi', 'basel', 'epub', 'vle', 'academics', 'institutional',
 'agency', 'doctoral', 'ell', 'novice', 'standardization', 'views','ideb', 'interface', 'sedmf', 'tools',
 'chat', 'uci', 'multilayer', 'programme', 'components', 'corpus', 'behaviours', 'sle', 'cuckoo', 'iop', 'generalizability', 'securityfirst',
 'pcp', 'radiograph', 'usepackage', 'pfa', 'modpfa', 'bdal', 'tam', 'pnm', 'feedback', 'catania', 'xapi', 'kyushu', 'syn', 'ecf', 'informatics',
 'monograph', 'languagestudent', 'clime', 'knn', 'kpis', 'pca', 'dnp', 'physical', 'rmodp', 'ministry', 'mlp', 'bookloan', 'esa', 'loop', 'uts',
 'thatstudent', 'text', 'mapreduce', 'repository', 'course', 'contribution', 'multilevel', 'england', 'services', 'ced', 'zayed',
 'exploratory','challenges', 'eiah', 'athabasca', 'ann', 'achievement', 'automl', 'raspberry', 'nii', 'ouj', 'irb','oman','architecture',
 'distance', 'japan', 'states', 'python', 'open', 'business', 'tutorial', 'technology','computer', 'coursera', 'american', 'academy', 'european',
 'project', 'discovery', 'engineering', 'world', 'information', 'cyber', 'machinery', 'teachinglearning_process',
 'methodology', 'malaysia', 'oman', 'base', 'switzerland', 'streams','digital', 'sciences', 'institute', 'computer_science', 'zealand',
 'taiwan']
print("the word number for PROPN removal: " + str(len(del_list_propn)))
data_lemmatized = remove_words(data_lemmatized, del_list_propn)



###################delete very high frequent words      
del_list_high_freq=['analytics', 'learning', 'analysis', 'quality', 'approaches', 'student', 'datum', 'analytic', 'education']
print("the word number for high_freq removal: " + str(len(del_list_high_freq)))
data_lemmatized = remove_words(data_lemmatized, del_list_high_freq)



################### delete country names      
del_list_country=['italian','pisa','union','saudi','africa','china','india','indian','united','state','unitedstate','spain','colombia','europe',
 'america','brazilian','african','brazil', 'germany', 'mexico','italian', 'oman' 'switzerland', 'california', 'france', 'arabia','norway','nigerian']
print("the word number for country names removal: " + str(len(del_list_country)))
data_lemmatized = remove_words(data_lemmatized, del_list_country)



################### delete words based on per document analysis with TFIDF      
del_list_per_document_tfidf=['silm', 'ppso', 'educloud', 'nigerian', 'spoc', 'uae', 'rnn', 'tvet', 'lambda', 'periodic', 'korean', 'admission', 'microlearne', 'tutorit', 'speak',
 'timeontask', 'geographic', 'metasynthesis', 'lap', 'envisage', 'boxing', 'liberal', 'reviews', 'fog', 'eventcentred', 'ssp', 'dataflow', 
 'sdn', 'softwaredefine', 'domainspecific', 'evidenceoflearne', 'documentation', 'xlearne', 'psp', 'schoolbag', 'boring', 'reactivity',
 'reactive', 'bat', 'proactive', 'diagram', 'sankey', 'crsra', 'near', 'supervise', 'analyzed', 'plot','finance', 'node', 'accumulate', 'independent',
 'collective', 'working', 'enforce','ide', 'none', 'formulate','increased', 'ceurws', 'unstructured', 'howstudent', 'demonstration', 'leadership', 
 'participate', 'ongoing', 'reduce','integrative', 'naver', 'limited','inherent', 'rationale', 'transition', 'handheld', 'probability', 'step',
 'short', 'thing', 'expertise', 'andstudent', 'let', 'pose', 'broaden', 'improved','whenstudent', 'involved', 'challenging', 'coefficient',
 'intuitive', 'adult', 'qualification', 'outside', 'rank', 'processing', 'side', 'workplace', 'profession', 'pace', 'diversity', 'merit',
 'learner', 'ling', 'mode', 'load', 'characterize', 'tostudent', 'diverse', 'script', 'aera', 'practice', 'notion', 'term', 'studentgenerate',
 'suggestion', 'live', 'reach', 'final', 'bothstudent', 'action', 'task', 'intelligent', 'implication', 'lifelong', 'component','delve','regime',
 'ple', 'aie', 'computation', 'organisation', 'intermediary', 'habit', 'exception', 'linguistic', 'actionable', 'participatory', 'dimensional',
 'eachstudent', 'principle', 'technique', 'undertake', 'body','e', 'awareness', 'corporate', 'controlled','researcher', 'instructional', 
 'teachinglearne', 'factor', 'exercise', 'continuous',  'opinion', 'cooccurrence', 'problem', 'expectation', 'unused', 'indicator', 'valuable',
 'construction', 'deep', 'opportunity', 'forstudent', 'multiple', 'cafe', 'program', 'maker', 'international', 'aboutstudent', 'openstudent',
 'assigned','practice','package','leading','employment', 'engagement','misconception','language']
print("the word number for country names removal: " + str(len(del_list_per_document_tfidf)))
data_lemmatized = remove_words(data_lemmatized, del_list_per_document_tfidf)


################### delete words for topic improvement (based on results)      
del_list_topic_improvement=['topic','model','decision','knowledge','information_system','data_analysis','application','analytical','technological',
 'interest','datadriven','competitive','teaching','platform','modelling','dataset','data_set','strategy','data_processe','database','preprocesse',
 'volume','experiment','factorization','predictive','visual','ontology','instrument','computersupporte','programming','application','social',
 'risk','teaching','solution','country','skill','block','portal','machine','linear','edtech','examiner','databased','traditional',
 'story','logs','ity','reduction','detection','empirical','filtering','matrix_factorization','graduate','feature_extraction','computationalmodele',
 'block','workflow','scientist','manager','voice','customize','motivate','active','apache_spark','stored','job',
 'pipeline','success','narrow', 'recognise','partial', 'sampling','studentrelate', 'subjective','separate', 'summarise','female','male','activitybase',
 'pervasive','cultivate','implications','illustrative','ngram','upcoming','modality', 'mine','immersive','inmemory','offtask','nonexpert',
 'white','elective','vulnerable','unexplored','desirable','designbase','binomial' ,'dataintensive','datainformed','curve','used',
 'usable','impossible','realworld','square','talent','sensitive','revolutionary','subsequent','ship', 'guideline','edu','emphasize','emergent',
 'highdimensional','homework','intellectual','shortterm','introductory','anticipate','clean','urban','physics','map','enhancement','flow','recurrent']


#,'higher_education'
#academic?,pattern??????, track????,class????, computer_aided_instruction??, online???, instructor???, 'electronic','monitor','experimental','learning_activity'

print("the word number for topic_improvement: " + str(len(del_list_topic_improvement)))
data_lemmatized = remove_words(data_lemmatized, del_list_topic_improvement)



################### Editing some features 
# replace some features
replace_dict = {
    "curriculumr" : "curriculum",
    "predictionor" : "prediction",
    "predictione" : "prediction",
    "predictionion" : "prediction",
    "predictionive" : "prediction",
    "selfregulate" : "self_regulated",
    "selfregulation" : "self_regulated",
    "online_learne" : "online_learning",
    'onlineonlinelearne' : "online_learning",
    'data_mine' : "data_mining",
    'machine_learne' : 'machine_learning',
    'decisionmake' : 'decision_making',
    'hadoopbase' : 'hadoop',
    'cloudbased': 'cloud',
    'lmss' :  'lms',
    'adaptivelearningenvironment' : 'adaptive_learning',
    'adaptive_learne' : 'adaptive_learning',
    'classroomlike' : 'classroom',
    'scaling' : 'scale',
    'modalities' : 'modality',
    'modalmodality' : 'modality',
    'egovernance' : 'governance',
    'inclassroom' : 'classroom',
    'multicultural' : 'culture',
    'behaviorism' : 'behaviorist',
    'eassessment' : 'assessment',
    'unsupervised_learning' : 'unsupervised_learne',
    'supervised_learning' : 'supervised_learne',
    'enrollment' : 'enrol',
    'enrolment' : 'enrol',
    'enrolled' : 'enrol',
    'lad' : 'dashboard',
    'ebookbase' : 'ebook',
    'threaten' : 'threat',
    'videoclickstream' : 'video clickstream',
    'technologyenhanced' : 'technology_enhanced_learning' 

}

#Word replace
rejoined_docs = []
for item in data_lemmatized:
    words = item
    j = " ".join(words)
    rejoined_docs.append(j)

rejoined_docs = (multiple_replace(replace_dict, rejoined_docs))    
data_lemmatized = list(sent_to_words(rejoined_docs))
#check:
#print(data_lemmatized[0])

