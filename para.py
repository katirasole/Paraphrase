from __future__ import division    # to calculate floting point division
import difflib
import fuzzyset
import Levenshtein
import nltk
import pyter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial import distance as dist
import string
import numpy as np
import math
import re
import pandas as pd

sentence1 = ('Here’s a quick look at why we can’t get enough and if you haven’t checked out Opal already, be sure to come by Fresh STORY to give it a go!')
sentence2 =('Here’s a quick look at why we can not be enough and if you have not already checked Opel, then make sure to come up with the latest story to make sure!')

#----------------------Clean sentences------------------------------------
def clean(text):
    
    text = ''.join(text)
    text = re.sub(r"['`,,_,’^&$€%[:!\-\"\\\/}{?\].]",'',text).strip()
    clean_text = re.sub(' +',' ',text) 
    
    return (clean_text)


#-------------diff------------------------------------------
def diff_similarity(sentence1, sentence):
    
    seq = difflib.SequenceMatcher(None, sentence1, sentence2)
    diff_sim = seq.ratio()
    
    return (diff_sim)


#-----------fuzzyset----------------------------------------
def fuzzyset_sim (sentence1, sentence2):
    
    a = fuzzyset.FuzzySet()
    a.add(sentence1)
    fuz_sim = a.get(sentence2)
    
    return (fuz_sim[0][0])

#--------------Levenshtein----------------------------------
def Leve(sentence1, sentence2):
    
    lev = Levenshtein.ratio(sentence1, sentence2)
    
    return (lev)


#------------Machine Translation Metrics (Bleu, TER)-----------------------------------
def machin_translation(sentence1, sentence2):
    
    b1 = clean(sentence1)
    b2 = clean(sentence2)
    
    a = b1.split()
    b = b2.split()
    
    bleu = nltk.translate.bleu_score.sentence_bleu([b1], b2, weights = (0.5, 0.5))
    #ter = pyter.ter(a, b)
    
    return(bleu)

#---------------distance metric (cosine similarity, chebychev)--------------------------------------------
def distance(sentence1, sentence2):
    
    text = []                                 # append two sentences into one to make tf_idf  matrix
    text.append(sentence1)
    text.append(sentence2)
       
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    
    a = (tfidf_matrix[0].todense())
    b = (tfidf_matrix[1].todense())
    

    cheb = dist.chebyshev(a, b)
    cosine = (cosine_similarity(a,b))[0][0]

    return(cheb, cosine)


#------------------------Matching Word(MW)--------------------------------
#Matching word ratio (MW) is a feature that indicates the similarity in terms of constituting words in sentences in given sentence pair.  
#The assumption behind this feature is that if two sentences have some words in common, they tend to be paraphrases of each other. 
#MW is calculated by dividing the number of words that occur in both sentences by the number of different words in sentence pair. 
#The feature gets its maximum value, 1, if sentences in pair hold exactly same words. The minimum MW value is zero in case where there is not a single word that is used in both sentences. 
#--------------------------------------------------------------------------

def matching_word(sentence1, sentence2):
    
    b1 = clean(sentence1)
    b2 = clean(sentence2)

    a = b1.split()
    b = b2.split()

    common = set(a).intersection(set(b) )
    common_arr =  list(common)
    unique = set(a).symmetric_difference(set(b) )
    unique_arr = list(unique)

    vocabulary = len(common) + len(unique)
    mw = len(common) / vocabulary

    return(mw)


#-----------------------Order of Word-------------------------------------------------------
#Word ordering ratio (OW) measures how similar the order of the words is in given sentences.
#In order to attain word-ordering ratio, for each common word in pair, the difference in word position, PD, is to be calculated. 
#For the words that are observed only in one of the sentences, PD value is accepted to be V where V is the total number of different words in pair.
#------------------------------------------------------------------------------------------

def word_ordering(sentence1, sentence2):
    
    pd_arr = []
    
    b1 = clean(sentence1)
    b2 = clean(sentence2)

    a = b1.split()
    b = b2.split()
    
    vocab = list(set(a + b))
    common = list(set(a).intersection( set(b) ))

    for i in range(len(common)):
        
        pos1 = a.index(common[i])
        pos2 = b.index(common[i])

        pd = abs(pos1-pos2)
        pd_arr.append(pd)
       
    for i in range (len(vocab) - len(common)):
        pd_arr.append(len(vocab))

    t = len(vocab)
    ow = 1 - ((sum(pd_arr)) / (t * t))

    return(ow)


#---------------------------------------------------------------------
diff = diff_similarity(sentence1, sentence2)
fuzzy = fuzzyset_sim(sentence1, sentence2)
lev = Leve(sentence1, sentence2)
bleu = machin_translation(sentence1, sentence2)
cheb, cosine = distance(sentence1, sentence2)
mw = matching_word (sentence1, sentence2)
ow = word_ordering (sentence1, sentence2)


raw_data_metric = { 'diff': diff , 'fuzzy': fuzzy, 'lev': lev, 'bleu': bleu,  'cheb': cheb, 'cosine': cosine, 'mw': mw, 'ow': ow}
print (raw_data_metric)
pd.DataFrame(raw_data_metric, columns = ['diff', 'fuzzy', 'lev', 'bleu', 'cheb', 'cosine', 'mw', 'ow' ], index=[0])
