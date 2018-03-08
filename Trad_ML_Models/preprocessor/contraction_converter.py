from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import WhitespaceTokenizer 
from nltk.corpus import wordnet
from expander import expand
from flattener import flatten

def token_punct_comma(tokens):
    for i in range(len(tokens)):
        if '.' in tokens[i]:
            tokenized = word_tokenize(tokens[i])
            tokens[i] = tokenized
    return tokens

def convert_sentence(sentence):
    tokens = WhitespaceTokenizer().tokenize(sentence.lower())
    expanded = expand(tokens)
    flattened = flatten(expanded)
    comma_tokenized = token_punct_comma(flattened)
    flattened2 = flatten(comma_tokenized)
    return flattened2

def contraction_converter(data):
    for line in data:
        line[1] = convert_sentence(line[1])
        line[2] = convert_sentence(line[2])
    return data
    
    

