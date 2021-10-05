import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, isdir, join


def preprocessing(train_path):
    files = listdir(train_path)
    #files.sort()

    #print(files)
    cnt = 0
    #print(len(files))
    TP, TN, FP, FN = 0, 0, 0, 0

    disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'nonobese', 'nondistended', 'non-distended', 'non-obese']


    df = pd.DataFrame()
    #df = pd.DataFrame(columns=['is_Obese', 'text_obese'] + disease)
    #df.columns = ['is_Obese', 'text_obese']

    for file in files:
        #if 'U' in file: continue
        f = open(join(train_path, file), 'r')
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
        #print(has_obes)
        if has_obes > 0:
            cnt += 1
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

        f.close()

    df.columns=['is_Obese', 'text_obese'] + disease
    df = df.reset_index(drop=True)
    
    return df

train_path = "Train_Textual/"
test_path = "Test_Intuitive/"
valid_path = "Validation/"

df_train = preprocessing(train_path)
df_test = preprocessing(test_path)
df_valid = preprocessing(valid_path)

df_train.to_csv('train_data.csv', index=None)
df_test.to_csv('test_data.csv', index=None)
df_valid.to_csv('valid_data.csv', index=None)

#print(df)
#print(cnt)
#print(TP, TN, FP, FN)
