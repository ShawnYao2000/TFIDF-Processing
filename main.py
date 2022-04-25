import os
import re
import numpy as np
import math
from nltk.stem.snowball import SnowballStemmer
from collections import Counter


def allfile(filepath):
    arr = []
    dir1 = os.listdir(filepath)
    for alldir1 in dir1:
        filepath2 = os.path.join(filepath, alldir1)
        dir2 = os.listdir(filepath2)
        for alldir2 in dir2:
            filepath3 = os.path.join(filepath2, alldir2)
            arr.append(filepath3)
    return arr


def wordinfile(word, cleanlst):
    filecount = 0
    for i in cleanlst:
        if word in i:
            filecount = filecount+1
    return filecount

def processed(cleanlst):
    list = []
    for t in cleanlst:
        for s in t:
            if s not in list:
                list.append(s)
    return list


def preprocess(file):
    with open('/Users/mac/Desktop/lab6-feature-generation/stopwords.txt', encoding='Latin1') as stopw:
        stop_words = set(stopw.read().split())
    text = open(file, 'r', encoding='Latin1').read()
    text1 = re.split(r'[^a-zA-Z]', text)
    text1 = list(filter(None, text1))
    filtered = []
    for w in text1:
        wd = re.sub(r'[^a-z]', '', w.lower()).strip()
        if wd not in stop_words and wd != '':
            filtered.append(wd)
    suffix_sentence = [stemmer.stem(wd) for wd in filtered]
    return suffix_sentence

def removewords(input):
    for i in range(len(input)):
        if list.__contains__(input[i]):
            input.remove(input[i])
            input.insert(-1, '')
    return input

# since above function replaced stopwords with '', the following content removed all ''
def removenull(input):
    input = [i for i in input if(len(str(i))!=0)]
    return input


filepath = '/Users/mac/Desktop/lab6-feature-generation/dataset'
stemmer = SnowballStemmer("english")

filelst = allfile(filepath)
N = len(filelst)

tfidflst = []
cleanlst = []

for file in filelst:
    cleanlst.append(preprocess(file))

corpuslst = processed(cleanlst)

for text in cleanlst:
    singletf = []
    sum = 0
    wordcount = len(text)
    dic = Counter(text)
    for i in corpuslst:
        if i not in text:
            tfidf = 0
        else:
            tf = dic[i] / wordcount
            idf = math.log(N / wordinfile(i,cleanlst), 10)
            tfidf = tf * idf
            sum = sum + math.pow(tfidf,2)
        singletf.append(tfidf)
    deno = math.pow(sum,0.5)
    for i, j in enumerate(singletf):
        if singletf[i] != 0:
            singletf[i] = j / deno
    tfidflst.append(singletf)

for i in range(len(tfidflst)):
    if tfidflst[i] == 0:
        tfidflst.remove(tfidflst[i])
        tfidflst.insert((-1, ''))
tfidflst = removenull(tfidflst)

matrix = np.column_stack((cleanlst,tfidflst))
np.savez('train-20ng.npz',X=matrix)