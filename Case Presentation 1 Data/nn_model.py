import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import f1_score

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

seed = 88
tf.random.set_seed(seed)
np.random.seed(seed)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#del_col = ['is_Obese', 'text_obese', 'obese', 'obesity']
del_col = ['is_Obese']


def find_useless_feature():
    global del_col
    df = pd.read_csv('train_data.csv')
    for col in df.columns:
        if col == 'is_Obese': continue
        if df[col].sum() == 0:
            del_col += [col]
            #print(col)
        if df[col].sum() == 400:
            del_col += [col]
            #print(col, 'fkkkkk')
    
def load_data(filename):
    
    df = pd.read_csv(filename)
        
    df = df.fillna(0)
    Y_df = df['is_Obese']
    if Y_df.dtype=='int64': Y_df = to_categorical(Y_df, 2)
    X_df = df.copy()
    X_df = X_df.drop(columns=del_col)

    X_df = X_df.astype('int64')
    print(len(Y_df))

    X_df = np.array(X_df)

    return X_df, Y_df
    
#print(X_train, Y_train)


def report_model(model, X_train, Y_train, X_test, Y_test, X_valid):

    train_pred = model.predict(X_train)
    print('Train')
    print(classification_report(Y_train, train_pred))
    print('Roc_Auc:', roc_auc_score(Y_train, train_pred))
    #print('f1_score:', f1_score(Y_train, train_pred))
    print('')

    test_pred = model.predict(X_test)
    print('Test')
    print(classification_report(Y_test, test_pred))
    print('Roc_Auc:', roc_auc_score(Y_test, test_pred))
    #print('f1_score:', f1_score(Y_test, test_pred))
    print('')
    
    valid_pred = model.predict(X_valid)
    #valid_pred = model.predict_proba(X_valid)[:,1]
    #print('Valid prediction:', valid_pred)
    pred_df = pd.DataFrame(data={'Filename': df_file_name, 'Obesity': valid_pred[:,1]})
    pred_df['Obesity'] = pred_df['Obesity'].astype('int64')
    pred_df.to_csv("./pred.csv", index=False)
    print(pred_df)
    print(pred_df['Obesity'].value_counts())

def train_nn_model(X_train, Y_train, X_test, Y_test, X_valid):
    model = Sequential()

    #layers = np.random.randint(7, 11)
    layers = 5
    for i in range(layers):
        #neurons = np.random.randint(300, 500)
        neurons = 1000
        model.add(Dense(units=neurons, activation='relu'))
        model.add(Dropout(0.3))
        


    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X_train, Y_train, batch_size=32, epochs=20)

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
    return pred_df

def report_nn_model(X_train, Y_train, X_test, Y_test, X_valid):
    results = pd.DataFrame(columns = ['Filename', 'Obesity'])
    result = train_nn_model(X_train, Y_train, X_test, Y_test, X_valid)
    results['Filename'] = result['Filename']
    results['Obesity'] = result['Obesity']
    #results['Obesity'] = results['Obesity'].astype('int64')
    
    cases = 30
    for i in range(cases):
        result = train_nn_model(X_train, Y_train, X_test, Y_test, X_valid)
        #result.to_csv("./preds/pred%d.csv"%i, index=False)
        print(results)
        results['Filename'] = result['Filename']
        results['Obesity'] += result['Obesity']

    #print(results)
    print(results['Obesity'].value_counts().sort_index())

    results['Obesity']/= cases+1
    #results['Obesity']+= 0.1
    results['Obesity'] = results['Obesity'].round().astype('int64')
    # results['Obesity'] = results['Obesity'].astype('int64')

    print(results)
    print(results['Obesity'].value_counts().sort_index())

    results.to_csv("./pred.csv", index=False)



if __name__ == '__main__':
    find_useless_feature()

    X_train, X_test, Y_train, Y_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()


    # valid(submittion)
    X_valid, df_file_name = load_data('valid_data.csv')


    # train model
    # use original train, test data to train
    X_train, Y_train = load_data('train_data.csv')
    X_test, Y_test = load_data('test_data.csv')



    # use merge(train+test, 428 datas) to train
    #X_merge_data, Y_merge_data = load_data('merge_data.csv')
    #X_train, X_test, Y_train, Y_test = train_test_split(X_merge_data, Y_merge_data, stratify=Y_merge_data, test_size=0.3, random_state=88)

    # X_train, Y_train = X_merge_data.copy(), Y_merge_data.copy()
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.3, random_state=88)



    model_rf = RandomForestClassifier(
         #                  criterion='entropy', max_depth=8, max_features=0.7, max_leaf_nodes=17,
          #                 min_samples_leaf=17, min_samples_split=30,
           #               min_weight_fraction_leaf=0.01, random_state=123
                            ).fit(X_train, Y_train)
    #model = LogisticRegression().fit(X_train, Y_train)

    result = train_nn_model(X_train, Y_train, X_test, Y_test, X_valid)
    #report_nn_model(X_train, Y_train, X_test, Y_test, X_valid)
    #report_model(model_rf, X_train, Y_train, X_test, Y_test, X_valid)
