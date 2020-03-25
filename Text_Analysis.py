#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:20:02 2020

@author: huangjinghua
"""

import pandas as pd
import re
import numpy as np
from PIL import Image
from collections import Counter
from wordcloud import WordCloud # using python 3.7
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# Create a string that contains all the information in the Description column
df = pd.read_csv('result.csv')
df = df.filter(['description'])
df = df.values.tolist()
text = []
for i in range (0, len(df)):
    text.append(df[i][0])

# Textual Analysis
STOPWORDS = ["an", "a", "the", "or", "and", "thou", "must", "that", "this", "self", "unless", "behind", "for", "which",
             "whose", "can", "else", "some", "will", "so", "from", "to", "by", "within", "of", "upon", "th", "with",
             "it"]

# Function that removes all stopwords in the text.
def _remove_stopwords(txt):
    """Delete from txt all words contained in STOPWORDS."""
    words = txt.split()
    # words = txt.split(" ")
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))

cleaned_text = []
for k in text:
    cleantextprep = str(k)
        # Regex cleaning
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  # apply regex
    cleantext = cleantextCAP.lower()  # lower case
    cleantext = _remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    cleaned_text.append(bound) 
    
# Split text into list of words
def decompose_word(doc):
    txt = []
    for word in doc:
        txt.extend(word.split())
    return txt

tokens = decompose_word(cleaned_text)

# Generate wordcloud
comment_words = ' '
for token in tokens:
    comment_words = comment_words + token + ' '
    
    
mask = np.array(Image.open("Cloud.png"))

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                mask=mask,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("wordcloud1.png",format='png',dpi=200)
plt.show()

# Read in negative lexicon
ndct = ''
with open('bl_negative.csv', 'r', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        ndct = ndct + line

# create a list of negative words
ndct = ndct.split('\n')
ndct = [entry for entry in ndct]
len(ndct)

# Read in positive lexicon
pdct = ''
with open('bl_positive.csv', 'r', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        pdct = pdct + line

pdct = pdct.split('\n')
pdct = [entry for entry in pdct]
len(pdct)

# Generate negative wordcloud
neg_words = ' '
for token in tokens:
    if token in ndct:
        neg_words = neg_words + token + ' '
    
wordcloud = WordCloud(background_color ='black',
                      mask=mask,
                      min_font_size = 10).generate(neg_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("neg_wordcloud.png",format='png',dpi=200)
plt.show()

# Generate positive wordcloud
pos_words = ' '
for token in tokens:
    if token in pdct:
        pos_words = pos_words + token + ' '
    
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                mask=mask,
                min_font_size = 10).generate(pos_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("pos_wordcloud.png",format='png',dpi=200)
plt.show()

# Number of words in article
nwords = len(tokens)

# Function that counts the number of positive and negative words that match the words in the lexicon.
def wordcount(words, dct):
    counting = Counter(words)
    count = []
    for key, value in counting.items():
        if key in dct:
            count.append([key, value])
    return count

# Number of negative words
nwc = wordcount(tokens, ndct)   # wordcount(text,lexicon)
# [['die', 3], ['famine', 1], ['lies', 2], ['foe', 1], ['cruel', 1], ['gaudy', 1], ['waste', 2], ['pity', 1], ['besiege', 1], ['tattered', 1], ['weed', 1], ['sunken', 1], ['shame', 3], ['excuse', 1], ['cold', 1], ['beguile', 1], ['wrinkles', 1], ['dies', 1], ['abuse', 1], ['deceive', 1], ['hideous', 1], ['sap', 1], ['frost', 1], ['prisoner', 1], ['bereft', 1], ['ragged', 1], ['forbidden', 1], ['death', 1], ['burning', 1], ['weary', 1], ['feeble', 1], ['sadly', 1], ['annoy', 1], ['offend', 1], ['chide', 1], ['wilt', 2], ['fear', 1], ['wail', 1], ['weep', 1], ['deny', 1], ['hate', 2], ['conspire', 1]]

# Number of positve words
pwc = wordcount(tokens, pdct)
# [['tender', 2], ['bright', 1], ['abundance', 1], ['sweet', 5], ['fresh', 2], ['spring', 1], ['proud', 1], ['worth', 1], ['beauty', 7], ['treasure', 3], ['praise', 2], ['fair', 3], ['proving', 1], ['warm', 1], ['fond', 1], ['lovely', 2], ['golden', 2], ['loveliness', 1], ['free', 1], ['beauteous', 2], ['great', 1], ['gentle', 2], ['work', 1], ['fairly', 1], ['excel', 1], ['leads', 1], ['willing', 1], ['happier', 2], ['gracious', 2], ['homage', 1], ['majesty', 1], ['heavenly', 1], ['strong', 1], ['adore', 1], ['like', 2], ['joy', 2], ['gladly', 1], ['pleasure', 1], ['sweetly', 1], ['happy', 1], ['pleasing', 1], ['well', 1], ['enjoys', 1], ['love', 4], ['beloved', 1]]

countries = open('Countries.txt', 'r')
list_countries = [line.split() for line in countries.readlines()]
List_countries = []
for i in range (0, len(list_countries)):
    List_countries.append(list_countries[i][0].lower())
List_Countries = []
for item in List_countries:
    List_Countries.append(item[1:-1])

country_count = wordcount(tokens, List_Countries)

# Total number of positive/negative words
ntot, ptot = 0, 0
for i in range(len(nwc)):
    ntot += nwc[i][1]

for i in range(len(pwc)):
    ptot += pwc[i][1]


# Print results
print('Positive words:')
for i in range(len(pwc)):
    print(str(pwc[i][0]) + ': ' + str(pwc[i][1]))
print('Total number of positive words: ' + str(ptot))
print('\n')
print('Percentage of positive words: ' + str(round(ptot / nwords, 4)))
print('\n')
print('Negative words:')
for i in range(len(nwc)):
    print(str(nwc[i][0]) + ': ' + str(nwc[i][1]))
print('Total number of negative words: ' + str(ntot))
print('\n')
print('Percentage of negative words: ' + str(round(ntot / nwords, 4)))
