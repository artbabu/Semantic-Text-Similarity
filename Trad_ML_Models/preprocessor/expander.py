
#Dictionary of contractions
contractions = {
  "'s": ["is"],
  "'re": ["are"],
  "aren't": ["are", "not"],
  "can't": ["can", "not"],
  "can't've": ["can", "not", "have"],
  "'cause": ["because"],
  "could've": ["could", "have"],
  "couldn't": ["could", "not"],
  "couldn't've": ["could", "not", "have"],
  "didn't": ["did", "not"],
  "doesn't": ["does", "not"],
  "don't": ["do", "not"],
  "hadn't": ["had", "not"],
  "hadn't've": ["had", "not", "have"],
  "hasn't": ["has", "not"],
  "haven't": ["have", "not"],
  "he'd": ["he", "would"],
  "he'd've": ["he", "would", "have"],
  "he'll": ["he", "will"],
  "he'll've": ["he", "will", "have"],
  "he's": ["he", "is"],
  "how'd": ["how", "did"],
  "how'd'y": ["how", "do", "you"],
  "how'll": ["how", "will"],
  "how's": ["how", "is"],
  "I'd": ["I would"],
  "I'd've": ["I", "would", "have"],
  "I'll": ["I", "will"],
  "I'll've": ["I", "will", "have"],
  "I'm": ["I", "am"],
  "I've": ["I", "have"],
  "isn't": ["is", "not"],
  "it'd": ["it", "had"],
  "it'd've": ["it", "would", "have"],
  "it'll": ["it", "will"],
  "it'll've": ["it", "will", "have"],
  "it's": ["it", "is"],
  "let's": ["let", "us"],
  "ma'am": ["madam"],
  "mayn't": ["may", "not"],
  "might've": ["might", "have"],
  "mightn't": ["might", "not"],
  "mightn't've": ["might", "not", "have"],
  "must've": ["must", "have"],
  "mustn't": ["must", "not"],
  "mustn't've": ["must", "not", "have"],
  "needn't": ["need", "not"],
  "needn't've": ["need", "not", "have"],
  "o'clock": ["of", "the", "clock"],
  "oughtn't": ["ought", "not"],
  "oughtn't've": ["ought", "not", "have"],
  "shan't": ["shall", "not"],
  "sha'n't": ["shall", "not"],
  "shan't've": ["shall", "not", "have"],
  "she'd": ["she", "would"],
  "she'd've": ["she", "would", "have"],
  "she'll": ["she", "will"],
  "she'll've": ["she", "shall", "have"],
  "she's": ["she", "is"],
  "should've": ["should", "have"],
  "shouldn't": ["should", "not"],
  "shouldn't've": ["should", "not", "have"],
  "so've": ["so", "have"],
  "so's": ["so", "as"],
  "that'd": ["that", "would"],
  "that'd've": ["that", "would", "have"],
  "that's": ["that", "has"],
  "there'd": ["there", "had"],
  "there'd've": ["there", "would", "have"],
  "there's": ["there", "is"],
  "they'd": ["they", "had "],
  "they'd've": ["they", "would", "have"],
  "they'll": ["they", "will"],
  "they'll've": ["they", "will", "have"],
  "they're": ["they", "are"],
  "they've": ["they", "have"],
  "to've": ["to", "have"],
  "wasn't": ["was", "not"],
  "we'd": ["we", "had"],
  "we'd've": ["we", "would", "have"],
  "we'll": ["we", "will"],
  "we'll've": ["we", "will", "have"],
  "we're": ["we", "are"],
  "we've": ["we", "have"],
  "weren't": ["were", "not"],
  "what'll": ["what", "will"],
  "what'll've": ["what", "will", "have"],
  "what're": ["what", "are"],
  "what's": ["what", "is"],
  "what've": ["what have"],
  "when's": ["when", "is"],
  "when've": ["when", "have"],
  "where'd": ["where", "did"],
  "where's": ["where", "is"],
  "where've": ["where", "have"],
  "who'll": ["who", "will"],
  "who'll've": ["who", "will", "have"],
  "who's": ["who", "is"],
  "who've": ["who", "have"],
  "why's": ["why", "is"],
  "why've": ["why", "have"],
  "will've": ["will", "have"],
  "won't": ["will", "not"],
  "won't've": ["will", "not", "have"],
  "would've": ["would", "have"],
  "wouldn't": ["would", "not"],
  "wouldn't've": ["would", "not", "have"],
  "y'all": ["you", "all"],
  "y'all'd": ["you", "all", "would"],
  "y'all'd've": ["you", "all", "would", "have"],
  "y'all're": ["you", "all", "are"],
  "y'all've": ["you", "all", "have"],
  "you'd": ["you", "had"],
  "you'd've": ["you", "would", "have"],
  "you'll": ["you", "will"],
  "you'll've": ["you", "will", "have"],
  "you're": ["you", "are"],
  "you've": ["you", "have"]
}

def expand_contraction(s, contractions_dict = contractions):
    return contractions_dict.get(s)

def find_contraction(s):
    if '\'' in s:
        return True
    return False

def replace_contraction(s):
    if(find_contraction(s)):
        expanded = expand_contraction(s)
    return expanded



def expand(tokens):
    for i in range(len(tokens)):
        if '\'' in tokens[i]:
            expandedtoken = replace_contraction(tokens[i])
            if(expandedtoken != None):
                tokens[i] = expandedtoken
    return tokens
    
        

