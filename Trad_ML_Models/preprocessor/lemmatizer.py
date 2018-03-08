from nltk import pos_tag
from nltk.corpus import wordnet as wn
from Flattener import flatten

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def tag_pos (sentence):
    tagged = pos_tag(sentence)
    return tagged

def replace_base (sentence):
    tagged = tag_pos(sentence)
    replaced = []
    for i in range(len(tagged)):
        token = tagged[i]
        word = token[0]
        pos = get_wordnet_pos(token[1])
        base = wn.morphy(word, pos)
        if(base != None):
            replaced.append(base)
        else:
            replaced.append(word)
    return replaced

def synonyms_word (word):
    wordlemma = []
    synsets = wn.synsets(word)
    for synset in synsets:
        lemma = [str(lemma.name()) for lemma in synset.lemmas()]
        wordlemma.append(lemma)
    flattened = flatten(wordlemma)
    unique = list(set(flattened))
    return unique

def synonyms_sentence (sentence):
    syn = []
    for word in sentence:
        syn.append(synonyms_word(word))
        
    return syn

def replace_syn(sentence1, sentence2):
    for i in range(len(sentence2)):
        a = synonyms_word(sentence2[i])
        for replacew in sentence1:
            b = synonyms_word(replacew)
            if(not set(a).isdisjoint(b)):
                sentence2[i] = replacew
                break
    return sentence1,sentence2


