import nltk
import numpy as np
import pandas as pd
import math
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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

#features = set()
#features.add('is_Obese')
#prev_27 = ['obesity', 'obese', 'bs', 'apnea', 'morbid', 'sleep', 'knee', 'right', 'gastric', 'levofloxacin', 'subcutaneously', 'renal','postoperative', 'bid', 'warfarin', 'continued', 'obstructive']
#disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'cholesterolemia', 'nondistended', 'non-distended']#, 'non-obese']
#prev_27 += disease

#for element in prev_27:
    
 #   element_lemmed = nltk_processing(element)
  #  features.add(element_lemmed[0])

#features.add('obesity', 'obese', 'bs', 'apnea', 'morbid', 'sleep', 'knee', 'right', 'gastric', 'levofloxacin', 'subcutaneously', 'renal','postoperative', 'bid', 'warfarin', 'continued', 'obstructive')



def generate_fixed_record(path, files):
    all_word_list = []
    if files == 0:
        files = listdir(path)
    #indexs = []
    #if 'Validation' in path: indexs = list(range(0,50))
    #else: indexs = list(range(0,400))
    
    #df = pd.DataFrame(columns=features, index = indexs)
    
    #print(df_1)
    cnt = 0
    for file in files:
        
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

        all_word_list.append(lemmed)
        
        #print(lemmed)
        #for item in lemmed:
         #   if len(item) > 3:
          #      all_word_list += item+' '

        cnt += 1
        
        #if cnt > 10: break
        #print(all_word_list)
        #if 'Validation' in path:
         #   f = open('valid_fixed/%s'%file, 'w')
          #  f.write(str(all_word_list))
           # f.close()
        #else:
         #   f = open('train_fixed/%s'%file, 'w')
          #  f.write(str(all_word_list))
           # f.close()
        #break
    
    #df = df.fillna(0)
    
    return all_word_list

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#pd.set_option("max_rows", 600)
from pathlib import Path  
import glob

def tf_idf(text_files,do_slice=False,terms=[]):
    #text_files =  glob.glob(f"{folder_path}/*.txt")
    num_files = len(text_files)
    #print(num_files)
    text_titles = [Path(text).stem for text in text_files]
    #print(text_titles)
    
    tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english') # 
    tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())
    #tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
    #tfidf_df = tfidf_df.sort_values(by = '00_Document Frequency', axis = 1, ascending = False)
    
    #tfidf_slice = tfidf_df[['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia',
    #                       'hypertension', 'hypothyroidism', 'gout', 'chronic']]
    if do_slice ==True:
        tfidf_slice = tfidf_df[terms]
        tfidf_slice.sort_index().round(decimals=2)
        return tfidf_slice,set(tfidf_vectorizer.get_feature_names()),tfidf_slice.describe()
    
    else:
        return tfidf_df,set(tfidf_vectorizer.get_feature_names()),tfidf_df.describe()
    
def show_meanRank(des_df,num):
    df_result = des_df.sort_values(by = 'mean', axis = 1, ascending = False)
    
    return df_result,df_result.columns[0:num]
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def tfidf_content(output,do_slice=False,terms=[]):
    tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english') # 
    tfidf_vector = tfidf_vectorizer.fit_transform(output)
    Index =list(range(0,len(output)))
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=Index, columns=tfidf_vectorizer.get_feature_names())
    
    if do_slice ==True:
        tfidf_slice = tfidf_df[terms]
        tfidf_slice.sort_index().round(decimals=2)
        return tfidf_slice,set(tfidf_vectorizer.get_feature_names()),tfidf_slice.describe()
    
    else:
        return tfidf_df,set(tfidf_vectorizer.get_feature_names()),tfidf_df.describe()


def generate_csv_file(words_list, test_words_list, valid_words_list):
    nltk_output = []
    test_nltk_output = []
    valid_nltk_output = []
    for T in words_list:
        nltk_output.append(' '.join(T))
    for T in test_words_list:
        test_nltk_output.append(' '.join(T)) 
    for T in valid_words_list:
        valid_nltk_output.append(' '.join(T))    
    Y_output = nltk_output[0:200]
    U_output = nltk_output[200:400]

    df_Y, Y_terms,des_Y = tfidf_content(Y_output)
    df_U, U_terms,des_U = tfidf_content(U_output)


    print('Num of Y terms:',len(Y_terms))
    print("Num of U terms:",len(U_terms))
        
    both_terms = Y_terms.intersection(U_terms)
    print('Num of Intersection terms:',len(both_terms))
    both_terms = sorted(both_terms)
    df_Y2, Y2_terms,des_Y2 = tfidf_content(Y_output,do_slice=True,terms=(both_terms))
    df_U2,U2_terms ,des_U2= tfidf_content(U_output,do_slice=True,terms=(both_terms))
        
    onlyY_terms = Y_terms - U_terms
    onlyY_terms = sorted(onlyY_terms)
    df_Y3, Y3_terms,des_Y3 = tfidf_content(Y_output,do_slice=True,terms=(onlyY_terms))
        
    #df_result3,ans2 = show_meanRank(des_Y3,20) #des_Y3.sort_values(by = 'mean', axis = 1, ascending = False)
    #print(ans2)
    #df_result3
        
    differ = des_Y2.values[1] - des_U2.values[1]
    differ = differ.reshape(1,len(differ))
    df_dif = pd.DataFrame(differ, index=['mean'], columns=both_terms)
    df_Results = df_dif.sort_values(by = 'mean', axis = 1, ascending = False)

    Results_terms = set(df_Results.columns[0:20])-set(['dilantin', 'pod', 'cyst', 'lmc'])
    print(len(Results_terms))
    print(Results_terms)

    df_Final, Final_terms,des_Final = tfidf_content(nltk_output,do_slice=True,terms=(Results_terms))
    df_Final_test, Final_terms_test,des_Final_test = tfidf_content(test_nltk_output,do_slice=True,terms=(Results_terms))
    df_T, T_terms,des_T = tfidf_content(valid_nltk_output,do_slice=True,terms=(Results_terms))

        
    df_Final['is_Obese'] = df_Final.index
    df_Final['is_Obese'] = np.where((df_Final['is_Obese']>=200), 1, 0)
    df_Final.to_csv('train_tfidf_data.csv', index=None)

 
    df_Final_test['is_Obese'] = df_Final_test.index
    df_Final_test['is_Obese'] = np.where((df_Final_test['is_Obese']>=200), 1, 0)
    df_Final_test.to_csv('test_tfidf_data.csv', index=None)

    import glob
    validText = glob.glob("Validation/*.txt")
    v_index = []
    for v in validText:
        v_index.append(v[-11:len(v)])
        # v_index.append(v.split("/")[-1])
    v_index.sort()
    df_T['is_Obese'] = v_index
    df_T.to_csv('valid_tfidf_data.csv', index=None)

def generate_csv_file_new(words_list, test_words_list, valid_words_list):
    nltk_output = []
    test_nltk_output = []
    valid_nltk_output = []
    for T in words_list:
        nltk_output.append(' '.join(T))
    for T in test_words_list:
        test_nltk_output.append(' '.join(T)) 
    for T in valid_words_list:
        valid_nltk_output.append(' '.join(T))    
    U_output = nltk_output[0:200]
    Y_output = nltk_output[200:400]

    df_Y, Y_terms,des_Y = tfidf_content(Y_output)
    df_U, U_terms,des_U = tfidf_content(U_output)

    #des_Y.sort_values(by = 'mean', axis = 1, ascending = False)

    print('Num of Y terms:',len(Y_terms))
    print("Num of U terms:",len(U_terms))
        
    both_terms = Y_terms.intersection(U_terms)
    print('Num of Intersection terms:',len(both_terms))
    both_terms = sorted(both_terms)
    df_Y2, Y2_terms,des_Y2 = tfidf_content(Y_output,do_slice=True,terms=(both_terms))
    df_U2,U2_terms ,des_U2= tfidf_content(U_output,do_slice=True,terms=(both_terms))
        
    onlyY_terms = Y_terms - U_terms
    onlyY_terms = sorted(onlyY_terms)
    df_Y3, Y3_terms,des_Y3 = tfidf_content(Y_output,do_slice=True,terms=(onlyY_terms))
        
    #df_result3,ans2 = show_meanRank(des_Y3,20) #des_Y3.sort_values(by = 'mean', axis = 1, ascending = False)
    #print(ans2)
    #df_result3
        
    differ = des_Y2.values[1] - des_U2.values[1]
    differ = differ.reshape(1,len(differ))
    df_dif = pd.DataFrame(differ, index=['mean'], columns=both_terms)
    df_Results = df_dif.sort_values(by = 'mean', axis = 1, ascending = False)

    Results_terms = set(df_Results.columns[0:20])-set(['dilantin', 'pod', 'cyst', 'lmc','nonischem','spong','fibromyalgia', 'noncardiac'])
    #Results_terms = Results_terms.union(set(['chronic','gout','hyperlipidemia',  'dyslipidemia',
                                              # 'hypertension', 'hypothyroidism','coronary', 'diabetic',
                                               # 'myocardial infarction', 'cholesterolemia','diabetes', 
                                             #'heart failure', 'non-distended', 'hypertriglyceridemia', 'nondistended'
     #                                        ]))
    #Results_terms = Results_terms.union(set([ 'creat',  'rouxeni', 'hypoventil', 'hypoxia',  
            #'collar','trach','pouch','porphyria','pl','hypokinet',
    #                                         ]))
    print(len(Results_terms))

    print(Results_terms)

    df_Final, Final_terms,des_Final = tfidf_content(nltk_output,do_slice=True,terms=(Results_terms))
    df_Final_test, Final_terms_test,des_Final_test = tfidf_content(test_nltk_output,do_slice=True,terms=(Results_terms))
    df_T, T_terms,des_T = tfidf_content(valid_nltk_output,do_slice=True,terms=(Results_terms))

    df_Final_test['is_Obese'] = df_Final_test.index
    df_Final_test['is_Obese'] = np.where((df_Final_test['is_Obese']>=200), 1, 0)
    df_Final_test.to_csv('test_tfidf_data.csv', index=None)
    
    df_Final['is_Obese'] = df_Final.index
    df_Final['is_Obese'] = np.where((df_Final['is_Obese']>=200), 1, 0)
    df_Final.to_csv('train_tfidf_data.csv', index=None)
    df_Final['is_Obese'].value_counts()

    import glob
    validText = glob.glob("Validation/*.txt")
    v_index = []
    for v in validText:
        v_index.append(v[-11:len(v)])
        #v_index.append(v.split('/')[-1])
    v_index.sort()
    df_T['is_Obese'] = v_index
    df_T.to_csv('valid_tfidf_data.csv', index=None)


if __name__ == '__main__':
    os.chdir('../Case Presentation 1 Data')

    train_path = "Train_Textual/"
    test_path = "Test_Intuitive/"
    valid_path = "Validation/"

    merge_path = "Merge_dataset/"

    
    
    #aa
    
    #preprocessing_to_find_feature(train_path, 0) # get features
    #print(df_train)
    
        
    words_list = []
    test_words_list = []
    valid_words_list = []

    try:
        with open('train_words_tfidf.data', 'rb') as filehandle:
            words_list = pickle.load(filehandle)
            
        with open('test_words_tfidf.data', 'rb') as filehandle:
            test_words_list = pickle.load(filehandle)

        with open('valid_words_tfidf.data', 'rb') as filehandle:
            valid_words_list = pickle.load(filehandle)
    except:
        words_list  = generate_fixed_record(train_path, 0) # generate fixed_txt in train_fixed
        test_words_list = generate_fixed_record(test_path, 0)
        valid_words_list = generate_fixed_record(valid_path, 0)

        #stop
        with open('train_words_tfidf.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(words_list, filehandle)

        with open('test_words_tfidf.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(test_words_list, filehandle)
            
        with open('valid_words_tfidf.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(valid_words_list, filehandle)
        

    generate_csv_file_new(words_list, test_words_list, valid_words_list)



    #Results_terms = use_tfidf_to_generate_train_csv_file()
    #use_tfidf_to_generate_valid_csv_file(Results_terms)
    

    

    #df_train = generate_csv_file('train_fixed/', 0)
    #df_valid = generate_csv_file('valid_fixed/', 0)
    #print(len(Y_terms),len(U_terms))
    #print(df_result.columns[0:30])    
    #df_test = preprocessing(test_path, 0)
    #df_valid = preprocessing(valid_path, 0)

    #df_train.to_csv('train_data.csv', index=None)
    #df_test.to_csv('test_data.csv', index=None)
    #df_valid.to_csv('valid_data.csv', index=None)

    

    #print(df_train)    
