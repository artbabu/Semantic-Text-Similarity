
# coding: utf-8

# In[ ]:


import itertools

def flatten(tokens):
    tokens2 = [([x] if isinstance(x,str) else x) for x in tokens]
    flattened = list(itertools.chain(*tokens2))
    return flattened

