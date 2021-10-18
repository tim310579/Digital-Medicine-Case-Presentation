import nltk
import numpy as np
import pandas as pd
import math
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
from os.path import isfile, isdir, join
import os
from string import digits
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_tokens(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    
    no_punctuation = lowers.translate(remove_punctuation_map)
    #print(no_punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


REMOVED_TITLES = ['POTENTIALLY SERIOUS INTERACTION', 'Discharge Date', 'Attending','Reason for override',
'DISCHARGE MEDICATIONS', 'Alert overridden', 'Dictated By', 'T', 'BRIEF RESUME OF HOSPITAL COURSE', 
'ENTERED BY', 'Service', 'DISCHARGE PATIENT ON', 'ALLERGIES', 'Override Notice', 'ALLERGY', 'CODE STATUS',
'Batch', 'CC', 'ATTENDING', 'eScription document', 'D', 'FAMILY HISTORY', 'MEDICATIONS', 
'Previous override information', 'SERIOUS INTERACTION', 'MEDICATIONS ON ADMISSION', 'CHLORIDE Reason for override', 
'ID', 'RETURN TO WORK', 'LISINOPRIL Reason for override', 'MEDICATIONS ON DISCHARGE', 'SERVICE', 'cc', 
'WARFARIN Reason for override', 'ADMISSION MEDICATIONS', 'Infectious disease', 'CURRENT MEDICATIONS', 
'Infectious Disease']

def reconstruct(text):
    '''
    To reconstruct the input article.
    Inputs:
        text: the original text
    Returns:
        text_rev: recontructed text
    '''

    text_rev = []
    text = text.replace('. ', '.\n')
    text = text.replace(',\n ', ', ')

    for line in text.split('\n'):
        if len(text_rev)==0 or ':' in line:
            text_rev.append(line)
        else:
            text_rev[-1] = text_rev[-1] + f' {line}'

    return '\n'.join(text_rev).replace(' , ', ', ')


def remove_section(text, titles=REMOVED_TITLES):
    '''
    To remove specified sections from the text.
    Inputs
        text, a string. The original text/article.
        titles, a list of string. A collection of section titles that will be removed from the original text/article.
    Returns
        text_rev, a string. The revised text/article.
    '''
    text = reconstruct(text)
    text_rev = []
    
    for section in text.split('\n'):
        relevent = True
        for t in titles:
            if (t+':') in section:
                relevent = False
                break
        
        if relevent:
            text_rev.append(section)
    
    return '\n'.join(text_rev).replace(' , ', ', ')

def nltk_processing(record):
    #print(record)
    lowers = record.lower()
    # remove punctuations
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)  
    no_punctuation = lowers.translate(remove_punctuation_map)
        
    tokens = get_tokens(lowers)
    # remove stopwords: the, a, this...
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    # remove ing, ed, es: brancing, branched, braanches-> branch
    stemmer = [PorterStemmer().stem(w) for w in filtered]
    # remove plural: apples->apple
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmer]

    return lemmed

features = set()
features.add('is_Obese')
prev_27 = ['obesity', 'obese', 'bs', 'apnea', 'morbid', 'sleep', 'knee', 'right', 'gastric', 'levofloxacin', 'subcutaneously', 'renal','postoperative', 'bid', 'warfarin', 'continued', 'obstructive']
disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'cholesterolemia', 'nondistended', 'non-distended']#, 'non-obese']
prev_27 += disease

for element in prev_27:
    
    element_lemmed = nltk_processing(element)
    features.add(element_lemmed[0])

#features.add('obesity', 'obese', 'bs', 'apnea', 'morbid', 'sleep', 'knee', 'right', 'gastric', 'levofloxacin', 'subcutaneously', 'renal','postoperative', 'bid', 'warfarin', 'continued', 'obstructive')


def preprocessing_to_find_feature(train_path, files):
    if files == 0:
        files = listdir(train_path)
    #files.sort()

    #disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'cholesterolemia', 'nondistended', 'non-distended']#, 'non-obese']
    #disease2 = ['obesity', 'obese', 'bs', 'apnea', 'morbid', 'sleep', 'knee', 'right', 'gastric', 'levofloxacin', 'subcutaneously', 'renal','postoperative', 'bid', 'warfarin', 'continued', 'obstructive'] 
    #disease_U = ['dehydration', 'osteopenia', 'renal dysfunction', 'hernia', 'angina', 'CHF', 'pulmonary edema', 'pneumothorax', 'osteoporosis', 'anemia', 'hematuria', 'arthritis']
    
    #disease += disease2
    #disease += disease_U
    
    df = pd.DataFrame()
    #df = pd.DataFrame(columns=['is_Obese', 'text_obese'] + disease)
    #df.columns = ['is_Obese', 'text_obese']
    # TP, FP, FN, TN = 0,0,0,0
    
    for file in files:
        #if 'U' in file: continue
        try:
            f = open(join(train_path, file), 'r')
        except:
            f = open(join(train_path, file.replace('U', 'N')), 'r')

        table = str.maketrans('', '', digits)
        
        records = f.read()
        records = reconstruct(records)
        records = remove_section(records, REMOVED_TITLES)
        
        record = records.translate(table)
        
        lemmed = nltk_processing(record)
        #print(lemmed)
        count = Counter(lemmed)
        
        for item in count.most_common(50):
            if item[1] > 2 and len(item[0]) > 2: features.add(item[0])
            #print(item[0])

        #break

    #return features



def count_feature_occurence(path, files):
    all_word_list = []
    if files == 0:
        files = listdir(path)
    indexs = []
    if 'Validation' in path: indexs = list(range(0,50))
    else: indexs = list(range(0,400))
    
    df = pd.DataFrame(columns=features, index = indexs)
    
    #print(df_1)
    cnt = 0
    for file in files:
        print(file)
        #if 'U' in file: continue
        try:
            f = open(join(path, file), 'r')
        except:
            f = open(join(path, file.replace('U', 'N')), 'r')

        table = str.maketrans('', '', digits)
        
        records = f.read()
        records = reconstruct(records)
        records = remove_section(records, REMOVED_TITLES)
        
        record = records.translate(table)

        lemmed = nltk_processing(record)
        
        #print(lemmed)
        all_word_list.append(lemmed)

        count = Counter(lemmed)
        #print(count.most_common(len(count)))

        #stop
        for item in count.most_common(len(count)):
            #print(item)
            if str(item[0]) in features:
                df.loc[cnt, str(item[0])] = item[1]
            
                
            #df.loc[cnt, str(item)] = no_punctuation.count(item)
            #print(item, no_punctuation.count(item))
            
        
        #df = pd.DataFrame(columns=['a','b'], index = [0,1])
        #df['a'] = 99
        if 'Valid' in path:
            df.loc[cnt, 'is_Obese'] = file
        else:
            if 'Y' in file:
                df.loc[cnt, 'is_Obese'] = 1
            else:
                df.loc[cnt, 'is_Obese'] = 0
        cnt += 1
        
        #if cnt > 10: break
    #print(all_word_list)
    #f = open('output.txt', 'w')
    #f.write(str(all_word_list))
    #f.close()
    
    
    df = df.fillna(0)
    
    return df, all_word_list
    
if __name__ == '__main__':
    os.chdir('../Case Presentation 1 Data')

    train_path = "Train_Textual/"
    test_path = "Test_Intuitive/"
    valid_path = "Validation/"

    merge_path = "Merge_dataset/"

    
    
    #aa

    #preprocessing_to_find_feature(train_path, 0) # get features
    #print(df_train)
    df_train, words_list = count_feature_occurence(train_path, 0)
    #df_test = count_feature_occurence(test_path, 0)
    #df_valid = count_feature_occurence(valid_path, 0)
    
    with open('to_words_for_tfidf.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(words_list, filehandle)



    #df_test = preprocessing(test_path, 0)
    #df_valid = preprocessing(valid_path, 0)

    df_train.to_csv('train_data.csv', index=None)
    #df_test.to_csv('test_data.csv', index=None)
    #df_valid.to_csv('valid_data.csv', index=None)

    print(df_train)    
