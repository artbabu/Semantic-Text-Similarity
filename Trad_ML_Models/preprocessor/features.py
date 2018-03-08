from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loader import *
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet_ic
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import MinMaxScaler

model = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)


testFeaturesFileName = 'testFeatures.csv'
trainFeaturesFileName = 'trainFeatures.csv'

testPreprocessedFileName = 'testPreprocessedData.csv'
trainPreprocessedFileName = 'trainPreprocessedData.csv'

cosSimFeatureCols = ['cos_sim']
lenFeatureCols = ['len_A', 'len_B', 'A_diff_B', 'B_diff_A',
                  'A_or_B', 'A_and_B', 'A_diff_B_div_len_B', 'B_diff_A_div_len_A',
                  'N_A_diff_B', 'V_A_diff_B', 'J_A_diff_B', 'R_A_diff_B',
                  'N_B_diff_A', 'V_B_diff_A', 'J_B_diff_A', 'R_B_diff_A']
tdFeatureCols = ['negDiff', 'A_ant_in_B', 'B_ant_in_A', 'contradiction',
                 'max_path_dSim', 'min_path_dSim', 'avg_path_dSim',
                 'max_lch_dSim', 'min_lch_dSim', 'avg_lch_dSim',
                 'max_wup_dSim', 'min_wup_dSim', 'avg_wup_dSim',
                 'max_jcn_dSim', 'min_jcn_dSim', 'avg_jcn_dSim']
simFeatureCols = ['max_path_sim', 'min_path_sim', 'avg_path_sim',
                 'max_lch_sim', 'min_lch_sim', 'avg_lch_sim',
                 'max_wup_sim', 'min_wup_sim', 'avg_wup_sim',
                 'max_jcn_sim', 'min_jcn_sim', 'avg_jcn_sim']
allFeatureCols = cosSimFeatureCols + lenFeatureCols + tdFeatureCols + simFeatureCols

def extractFeatures(pairs, cols = allFeatureCols, scale=True):
    features = pairs[cols]
    if (scale):
        scaler = MinMaxScaler(feature_range=(-1, 1))    
        return scaler.fit_transform(features)
    else:
        return features

def extractTarget(pairs):   
    return pairs['relatedness_score']

def convert2BOWV(word_list):
    wv_list = []
    
    for w in word_list:
        if w in model.vocab:
            wv_list.append(model[w]) 
    return wv_list

# sum of all word vectors
def getSentenceEmbedings(word_list):
    wv_list = convert2BOWV(word_list)
    sv = wv_list[0]
    for i in range(len(wv_list)):
        if i != 0:
            sv = np.add(sv,wv_list[i])
    return sv.reshape(1, -1) 

def cosineSimFeatures(pairs):  
    pairs['sv_A'] = pairs['sentence_A'].map(getSentenceEmbedings)
    pairs['sv_B'] = pairs['sentence_B'].map(getSentenceEmbedings)		
    pairs['cos_sim'] = pairs.apply(lambda x: cosine_similarity(x['sv_A'], x['sv_B']).item(0), axis=1)
    
    #Drop temp cols
    dropCols = ['sv_A', 'sv_B']
    for d in dropCols:
        pairs.drop(d, axis=1, inplace=True)

def separateTags(tags):
    pos = {'N': set(), 'V': set(), 'J': set(), 'R': set()}
    for t in tags:
        #Just get what the tag starts with 
        (word, tag) = t
        if(tag[0] in ['N', 'V', 'J', 'R']):
            pos[tag[0]].add(word)
    return pos


def lengthFeatures(pairs, aTags, bTags):
    posDF = pd.DataFrame() 
    posDF['pos_A'] = aTags.map(separateTags)
    posDF['pos_B'] = bTags.map(separateTags)
    
    pairs['len_A'] = pairs['sentence_A'].map(len)
    pairs['len_B'] = pairs['sentence_B'].map(len)
    
    pairs['A_diff_B'] = (pairs['sentence_A'] - pairs['sentence_B']).map(len)
    pairs['B_diff_A'] = (pairs['sentence_B'] - pairs['sentence_A']).map(len)
    
    pairs['A_or_B'] =  pairs.apply(lambda x: x['sentence_A'] | x['sentence_B'], axis=1).map(len)
    pairs['A_and_B'] = pairs.apply(lambda x: x['sentence_A'] & x['sentence_B'], axis=1).map(len)
    
    pairs['A_diff_B_div_len_B'] = pairs['A_diff_B'] / pairs['len_B'].map(float)
    pairs['B_diff_A_div_len_A'] = pairs['B_diff_A'] / pairs['len_A'].map(float)
    
    pairs['N_A_diff_B'] =  posDF.apply(lambda x: x['pos_A']['N'] - x['pos_B']['N'], axis=1).map(len)
    pairs['V_A_diff_B'] =  posDF.apply(lambda x: x['pos_A']['V'] - x['pos_B']['V'], axis=1).map(len)
    pairs['J_A_diff_B'] =  posDF.apply(lambda x: x['pos_A']['J'] - x['pos_B']['J'], axis=1).map(len)
    pairs['R_A_diff_B'] =  posDF.apply(lambda x: x['pos_A']['R'] - x['pos_B']['R'], axis=1).map(len)
    
    pairs['N_B_diff_A'] =  posDF.apply(lambda x: x['pos_B']['N'] - x['pos_A']['N'], axis=1).map(len)
    pairs['V_B_diff_A'] =  posDF.apply(lambda x: x['pos_B']['V'] - x['pos_A']['V'], axis=1).map(len)
    pairs['J_B_diff_A'] =  posDF.apply(lambda x: x['pos_B']['J'] - x['pos_A']['J'], axis=1).map(len)
    pairs['R_B_diff_A'] =  posDF.apply(lambda x: x['pos_B']['R'] - x['pos_A']['R'], axis=1).map(len)

    del posDF


def aAntonymsInb(aSynsets, b):
    aAntInb = set()                 
    for syn in aSynsets:
        for l in syn.lemmas():
            if l.antonyms() and (l.antonyms()[0].name() in b):
                aAntInb.add(l.antonyms()[0].name())
    return aAntInb

def pathSim(a, b):
    sim = []
    for s1 in a:
        for s2 in b: 
            path_sim = s1.path_similarity(s2)
            if path_sim:
                sim.append(path_sim)
    return sim

def lchSim(a, b):
    sim = []
    for s1 in a:
        for s2 in b: 
            try:
                lch_sim = s1.lch_similarity(s2)
                if lch_sim:
                    sim.append(lch_sim)
            except:
                pass
    return sim

def wupSim(a, b):
    sim = []
    for s1 in a:
        for s2 in b: 
            wup_sim = s1.wup_similarity(s2)
            if wup_sim:
                sim.append(wup_sim)
    return sim

inf = 1e300
inf_replacement = 6
def jcnSim(a, b, wnic):
    sim = []
    for s1 in a:
        for s2 in b: 
            try:
                jcn_sim = s1.jcn_similarity(s2, wnic)
                if jcn_sim :
                    sim.append(jcn_sim if abs(jcn_sim) < inf else inf_replacement )
            except:
                pass
    return sim


def contradictionFeatures(pairs, f1Tags, f2Tags):
    negationWords = {'never', 'no', 'nothing', 'nowhere', 'none', 'not', 'n\'t'}
    
    #Negative words in each sentence
    pairs['negDiff'] = pairs.apply(lambda x: 1 if (len(x['sentence_A'] & negationWords) != len(x['sentence_B'] & negationWords)) else -1, axis=1)
    #pairs['neg_B'] = pairs.apply(lambda x: x['sentence_B'] & negationWords, axis=1).map(len)
    
    pairs['synset_A'] = f1Tags.map(getSynset)
    pairs['synset_B'] = f2Tags.map(getSynset)

    pairs['A_ant_in_B'] = pairs.apply(lambda x: aAntonymsInb(x['synset_A'], x['sentence_B']), axis=1).map(len)
    pairs['B_ant_in_A'] = pairs.apply(lambda x: aAntonymsInb(x['synset_B'], x['sentence_A']), axis=1).map(len)

    #pairs['negDiff'] = 1 if np.any([neg_A != neg_B]) else -1
    pairs['contradiction'] = pairs.apply(lambda x: 1 if (x['negDiff'] or x['A_ant_in_B'] > 0 or x['B_ant_in_A'] > 0) else -1 , axis=1)
    #1 if np.any([(neg_A != neg_B), (A_ant_in_B > 0), (B_ant_in_A > 0)]) else -1

def similarityFeatures(pairs):
    # pairs['SA_diff_SB'] = (pairs['synset_A'] - pairs['synset_B'])
    # pairs['SB_diff_SA'] = (pairs['synset_B'] - pairs['synset_A'])
    # Avoid div by 0
    pairs['SB_SA_len'] = pairs.apply(lambda x: (len(x['synset_A']) * len(x['synset_B']))
    if np.all([len(x['synset_A']), len(x['synset_B'])]) else 1, axis=1).map(float)
    # pairs['SB_SA_diff_len'] =  pairs.apply(lambda x: (len(x['SA_diff_SB']) * len(x['SB_diff_SA']))
    # if np.all([len(x['SA_diff_SB']), len(x['SB_diff_SA'])]) else 1, axis=1).map(float)

    wnic = wordnet_ic.ic('ic-brown.dat')

    pairs['raw_path_sim'] = pairs.apply(lambda x: pathSim(x['synset_A'], x['synset_B']), axis=1)
    pairs['max_path_sim'] = pairs['raw_path_sim'].map(lambda x: max(x) if x else 0)
    pairs['min_path_sim'] = pairs['raw_path_sim'].map(lambda x: min(x) if x else 0)
    pairs['avg_path_sim'] = pairs['raw_path_sim'].map(sum) / pairs['SB_SA_len']

    pairs['raw_lch_sim'] = pairs.apply(lambda x: lchSim(x['synset_A'], x['synset_B']), axis=1)
    pairs['max_lch_sim'] = pairs['raw_lch_sim'].map(lambda x: max(x) if x else 0)
    pairs['min_lch_sim'] = pairs['raw_lch_sim'].map(lambda x: min(x) if x else 0)
    pairs['avg_lch_sim'] = pairs['raw_lch_sim'].map(sum) / pairs['SB_SA_len']

    pairs['raw_wup_sim'] = pairs.apply(lambda x: wupSim(x['synset_A'], x['synset_B']), axis=1)
    pairs['max_wup_sim'] = pairs['raw_wup_sim'].map(lambda x: max(x) if x else 0)
    pairs['min_wup_sim'] = pairs['raw_wup_sim'].map(lambda x: min(x) if x else 0)
    pairs['avg_wup_sim'] = pairs['raw_wup_sim'].map(sum) / pairs['SB_SA_len']

    pairs['raw_jcn_sim'] = pairs.apply(lambda x: jcnSim(x['synset_A'], x['synset_B'], wnic), axis=1)
    pairs['max_jcn_sim'] = pairs['raw_jcn_sim'].map(lambda x: max(x) if x else 0)
    pairs['min_jcn_sim'] = pairs['raw_jcn_sim'].map(lambda x: min(x) if x else 0)
    pairs['avg_jcn_sim'] = pairs['raw_jcn_sim'].map(sum) / pairs['SB_SA_len']

    # Drop temp cols
    """
    dropCols = ['raw_path_sim', 'raw_lch_sim', 'raw_wup_sim', 'raw_jcn_sim', 'SA_diff_SB', 'SB_diff_SA', 'SB_SA_len']
    dropCols = ['raw_path_sim', 'raw_lch_sim', 'raw_wup_sim', 'raw_jcn_sim', 'SA_diff_SB', 'SB_diff_SA', 'SB_SA_len']
    for d in dropCols:
        pairs.drop(d, axis=1, inplace=True)
    """
def similaritDiffFeatures(pairs):
    pairs['SA_diff_SB'] = (pairs['synset_A'] - pairs['synset_B'])
    pairs['SB_diff_SA'] = (pairs['synset_B'] - pairs['synset_A'])
    #Avoid div by 0
    pairs['SB_SA_diff_len'] =  pairs.apply(lambda x: (len(x['SA_diff_SB']) * len(x['SB_diff_SA']))
		    if np.all([len(x['SA_diff_SB']), len(x['SB_diff_SA'])]) else 1, axis=1).map(float)

    #wnic = wordnet_ic.ic('ic-brown-resnik-add1.dat')
    wnic = wordnet_ic.ic('ic-brown.dat')

    pairs['raw_path_dSim'] = pairs.apply(lambda x: pathSim(x['SA_diff_SB'], x['SB_diff_SA']), axis=1)
    pairs['max_path_dSim'] = pairs['raw_path_dSim'].map(lambda x: max(x) if x else 0)
    pairs['min_path_dSim'] = pairs['raw_path_dSim'].map(lambda x: min(x) if x else 0)
    pairs['avg_path_dSim'] = pairs['raw_path_dSim'].map(sum) / pairs['SB_SA_diff_len']

    pairs['raw_lch_dSim'] = pairs.apply(lambda x: lchSim(x['SA_diff_SB'], x['SB_diff_SA']), axis=1)
    pairs['max_lch_dSim'] = pairs['raw_lch_dSim'].map(lambda x: max(x) if x else 0)
    pairs['min_lch_dSim'] = pairs['raw_lch_dSim'].map(lambda x: min(x) if x else 0)
    pairs['avg_lch_dSim'] = pairs['raw_lch_dSim'].map(sum) / pairs['SB_SA_diff_len']

    pairs['raw_wup_dSim'] = pairs.apply(lambda x: wupSim(x['SA_diff_SB'], x['SB_diff_SA']), axis=1)
    pairs['max_wup_dSim'] = pairs['raw_wup_dSim'].map(lambda x: max(x) if x else 0)
    pairs['min_wup_dSim'] = pairs['raw_wup_dSim'].map(lambda x: min(x) if x else 0)
    pairs['avg_wup_dSim'] = pairs['raw_wup_dSim'].map(sum) / pairs['SB_SA_diff_len']

    pairs['raw_jcn_dSim'] = pairs.apply(lambda x: jcnSim(x['SA_diff_SB'], x['SB_diff_SA'], wnic), axis=1)
    pairs['max_jcn_dSim'] = pairs['raw_jcn_dSim'].map(lambda x: max(x) if x else 0)
    pairs['min_jcn_dSim'] = pairs['raw_jcn_dSim'].map(lambda x: min(x) if x else 0)
    pairs['avg_jcn_dSim'] = pairs['raw_jcn_dSim'].map(sum) / pairs['SB_SA_diff_len']

    #Drop temp cols
    """
    dropCols = ['raw_path_sim', 'raw_lch_sim', 'raw_wup_sim', 'raw_jcn_sim', 'SA_diff_SB', 'SB_diff_SA', 'SB_SA_len']
    dropCols = ['raw_path_sim', 'raw_lch_sim', 'raw_wup_sim', 'raw_jcn_sim', 'SA_diff_SB', 'SB_diff_SA', 'SB_SA_len']
    for d in dropCols:
        pairs.drop(d, axis=1, inplace=True)
    """

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def tagged_to_synset(wordTag):
    word, tag = wordTag
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        print("{} to {}".format(word, wn.synsets(word, wn_tag)[0]))
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def getSynset(tag):
    synset = set()
    for t in tag:
        word, tag = t
        wn_tag = penn_to_wn(tag)
        if wn_tag is None:
            continue
        try:
            synset.add(wn.synsets(word, wn_tag)[0])
        except:
            continue
    return synset

def loadFeatures(pairs, featuresOutputFileName):
    pairs['sentence_A'] = pairs['sentence_A'].map(lambda x: x.split())
    pairs['sentence_B'] = pairs['sentence_B'].map(lambda x: x.split())
    
    #Cosine Similarity
    cosineSimFeatures(pairs)
    
    pairs['sentence_A'] = pairs['sentence_A'].map(set)
    pairs['sentence_B'] = pairs['sentence_B'].map(set)
    
    #Get tags
    f1Tags = pairs['sentence_A'].map(pos_tag)
    f2Tags = pairs['sentence_B'].map(pos_tag)
        
    #3.2.1 Length Features (len)
    #length information
    lengthFeatures(pairs, f1Tags, f2Tags)

    #3.2.5 Text Difference Measures (td)
    #Contradiction Entailment Relationship
    contradictionFeatures(pairs, f1Tags, f2Tags)
            
    #Similarity measures (i.e., path, lch, wup, jcn) of A-B and B-A
    similaritDiffFeatures(pairs)
    similarityFeatures(pairs)
  
    #Drop temp cols
    dropCols = ['synset_A', 'synset_B']
    """
    for d in dropCols:
        pairs.drop(d, axis=1, inplace=True)
    """

    #Save to csv
    pairs['sentence_A'] = pairs['sentence_A'].map(lambda x: " ".join(x))
    pairs['sentence_B'] = pairs['sentence_B'].map(lambda x: " ".join(x))
    pairs.to_csv(featuresOutputFileName, index=False)


def getTestFeatures(refresh = False):
    return getFeatures(testFeaturesFileName, testPreprocessedFileName, refresh)

def getTrainFeatures(refresh = False):
    return getFeatures(trainFeaturesFileName, trainPreprocessedFileName, refresh)

def getFeatures(featuresFileName, preprocessedFileName, refresh):
    curfilePath = os.path.abspath('__file__')
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    featuresDataPath = os.path.join(parentDir, 'data', featuresFileName)
    preprocessedDataPath = os.path.join(parentDir, 'data', preprocessedFileName)

    if not refresh and os.path.isfile(featuresDataPath) :
        print("Loading existing features. NOT Recalculating...")
        features = pd.read_csv(featuresDataPath)
        features['sentence_A'] = features['sentence_A'].map(lambda x: x.split())
        features['sentence_B'] = features['sentence_B'].map(lambda x: x.split())
        return features
    else:
        print("No existing features. Recalculating...")
        pairs = pd.read_csv(preprocessedDataPath)
        loadFeatures(pairs, featuresDataPath)
        return pairs

def test():
    getTestFeatures(True)
	
    #print("All columns: {}".format(pairs.head(0)))
    #print("extractFeatures: {}".format(extractFeatures(pairs)))

