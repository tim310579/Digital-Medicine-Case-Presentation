import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, isdir, join


def preprocessing(train_path):
    files = listdir(train_path)
    files.sort()

    #print(files)
    cnt = 0
    #print(len(files))
    TP, TN, FP, FN = 0, 0, 0, 0

    disease = ['coronary', 'diabetic', 'diabetes', 'hypertriglyceridemia', 'dyslipidemia', 'hypertension', 'hypothyroidism', 'Hyperlipidemia', 'gout', 'chronic', 'myocardial infarction', 'heart failure', 'non-distended', 'cholesterolemia', 'nondistended', 'non-distended', 'non-obese']


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

train_path = "../input/digital/Train_Textual/"
test_path = "../input/digital/Test_Intuitive"
valid_path = "../input/digital/Validation"

df_train = preprocessing(train_path)
df_test = preprocessing(test_path)
df_valid = preprocessing(valid_path)

df_train.to_csv('train_data.csv', index=None)
df_test.to_csv('test_data.csv', index=None)
df_valid.to_csv('valid_data.csv', index=None)

#print(df)
#print(cnt)
#print(TP, TN, FP, FN)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

del_col = ['is_Obese', 'text_obese']
#del_col = ['is_Obese']

df = pd.read_csv('train_data.csv')
#print(df)

Y_train = df['is_Obese']
X_train = df
X_train = X_train.drop(columns=del_col)
#print(X_train, Y_train)

df_test = pd.read_csv('test_data.csv')

Y_test = df_test['is_Obese']
X_test = df_test
X_test = X_test.drop(columns=del_col)

df_valid = pd.read_csv('valid_data.csv')

df_file_name = df_valid['is_Obese']
X_valid = df_valid
X_valid = X_valid.drop(columns=del_col)

from sklearn.metrics import f1_score
def report_model(model, X_train, Y_train, X_test, Y_test, X_valid):

    train_pred = model.predict(X_train)
    print('Train')
    print(classification_report(Y_train, train_pred))
    print('Roc_Auc:', roc_auc_score(Y_train, train_pred))
    print('f1_score:', f1_score(Y_train, train_pred))
    print('')

    test_pred = model.predict(X_test)
    print('Test')
    print(classification_report(Y_test, test_pred))
    print('Roc_Auc:', roc_auc_score(Y_test, test_pred))
    print('f1_score:', f1_score(Y_test, test_pred))
    print('')
    
    valid_pred = model.predict(X_valid)
    #valid_pred = model.predict_proba(X_valid)[:,1]
    print('Valid prediction:', valid_pred)
    pred_df = pd.DataFrame(data={'Filename': df_file_name, 'Obesity': valid_pred})
    pred_df.to_csv("./pred.csv", index=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#model = DecisionTreeClassifier().fit(X_train, Y_train)
model = RandomForestClassifier(
                        #class_weight={0:1,1:5}
#                       criterion='entropy', max_depth=8, max_features=0.7, max_leaf_nodes=17,
 #                      min_samples_leaf=17, min_samples_split=30,
  #                    min_weight_fraction_leaf=0.01, random_state=123
                        ).fit(X_train, Y_train)
#model = LogisticRegression().fit(X_train, Y_train)

#report_model(model, X_train, Y_train, X_test, Y_test, X_valid)
#np.set_printoptions(suppress=True)
#print('feature_importances_:', model.feature_importances_)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.fc1 = nn.Linear(17, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

net = net()

print(net)
output = net(X_train)

print(output)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

model = Sequential()

model.add(Dense(units=600, activation='relu'))

model.add(Dense(units=600, activation='relu'))
model.add(Dense(units=600, activation='relu'))
model.add(Dense(units=600, activation='relu'))

model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
# train model
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)
model.fit(X_train, Y_train, batch_size=50, epochs=50)

# evaluate the model and output the accuracy
result_train = model.evaluate(X_train, Y_train)
result_test = model.evaluate(X_test, Y_test)
   
print('Train Acc:', result_train[1])
print('Test Acc:', result_test[1])
predict = model.predict(X_valid)

pred_df = pd.DataFrame(data={'Filename': df_file_name, 'Obesity': np.round(predict)[:,1]})
pred_df['Obesity'] = pred_df['Obesity'].astype('int64')
pred_df.to_csv("./pred.csv", index=False)
print(pred_df)
print(pred_df['Obesity'].value_counts())


