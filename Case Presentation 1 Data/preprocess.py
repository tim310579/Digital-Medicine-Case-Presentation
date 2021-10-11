import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, isdir, join
import os

def preprocessing(train_path, files):
    if files == 0:
        files = listdir(train_path)
    #files.sort()

    disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'cholesterolemia', 'nondistended', 'non-distended']#, 'non-obese']


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
        record = f.read()
        has_obes = record.upper().count('obes'.upper()) + record.upper().count('overweight'.upper())
        
        has_disease = []
        for item in disease:
            has_disease.append(record.upper().count(item.upper()))
        #print(has_obes)
        #print([['1', has_obes] + has_disease])
        if 'Valid' in train_path:
            df = df.append([[file, has_obes] + has_disease])
        else:
            if 'Y' in file:
                df = df.append([[1, has_obes] + has_disease])
            else:
                df = df.append([[0, has_obes] + has_disease])
        '''
        if has_obes > 0:
            if 'Y' in file: 
                #print(file, has_obes)
                TP += 1
            else: 
                FP += 1
            #print(file, has_obes, has_disease)
        else:
            if 'Y' in file: 
                FN += 1
                #print(file, has_obes, has_disease)
            else: 
                TN += 1
            #print(file, has_obes, has_disease)
        '''
        f.close()

    df.columns=['is_Obese', 'text_obese'] + disease
    df = df.reset_index(drop=True)
    #print(TP, FP, FN, TN)
    
    return df


def merge_train_test_dir(train_path, test_path):
    

    files_train = listdir(train_path)
    files_test = listdir(test_path)

    
    for i in range(len(files_test)):
        files_test[i] = files_test[i].replace('N', 'U')
        
    all_files = files_train + files_test

    set_tmp = set(all_files) #use set to remove duplicated
    #print(len(set_tmp))
    useless_record = ['ID_716', 'ID_725', 'ID_728', 'ID_737', 'ID_740', 'ID_747', 'ID_851', 'ID_855', 'ID_861', 'ID_869', 'ID_882', 'ID_884', 'ID_891', 'ID_715', 'ID_726', 'ID_734', 'ID_739', 'ID_746', 'ID_750', 'ID_854', 'ID_857', 'ID_868', 'ID_873', 'ID_876', 'ID_883', 'ID_890', 'ID_892', 'ID_897', 'ID_904', 'ID_909', 'ID_915', 'ID_921', 'ID_929', 'ID_932', 'ID_935', 'ID_943', 'ID_945']

    for item in useless_record:
        set_tmp.remove('U_'+item+'.txt')
        set_tmp.remove('Y_'+item+'.txt')

    print(len(set_tmp))
    #print(set_tmp)

    return list(set_tmp)
    
train_path = "Train_Textual/"
test_path = "Test_Intuitive/"
valid_path = "Validation/"

merge_path = "Merge_dataset/"

files_merge = merge_train_test_dir(train_path, test_path)
#print(files_merge)

df_merge = preprocessing(merge_path, files_merge)
df_merge.to_csv('merge_data.csv', index=None)
#aa


df_train = preprocessing(train_path, 0)
df_test = preprocessing(test_path, 0)
df_valid = preprocessing(valid_path, 0)

df_train.to_csv('train_data.csv', index=None)
df_test.to_csv('test_data.csv', index=None)
df_valid.to_csv('valid_data.csv', index=None)

#print(df)
#print(cnt)
#print(TP, TN, FP, FN)
