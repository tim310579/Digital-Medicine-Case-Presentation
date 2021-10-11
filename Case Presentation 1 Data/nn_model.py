import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os

del_col = ['is_Obese', 'text_obese']
#del_col = ['is_Obese']

def load_data(filename):
    
    df = pd.read_csv(filename)
    #print(df)

    Y_df = df['is_Obese']
    if Y_df.dtype=='int64': Y_df = to_categorical(Y_df, 2)
    X_df = df
    X_df = X_df.drop(columns=del_col)

    return X_df, Y_df
    
#print(X_train, Y_train)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical



X_train, X_test, Y_train, Y_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()


# valid(submittion)
X_valid, df_file_name = load_data('valid_data.csv')

# train model
# use original train, test data to train
X_train, Y_train = load_data('train_data.csv')
X_test, Y_test = load_data('test_data.csv')



# use merge(train+test, 428 datas) to train
# X_merge_data, Y_merge_data = load_data('merge_data.csv')
# X_train, X_test, Y_train, Y_test = train_test_split(X_merge_data, Y_merge_data, stratify=Y_merge_data, test_size=0.3, random_state=88)

# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.3, random_state=88)



def train_nn_model():
    model = Sequential()

    layers = np.random.randint(3, 6)
    for i in range(layers):
        neurons = np.random.randint(300, 500)
        model.add(Dense(units=neurons, activation='relu'))
    # model.add(Dropout(0.5))
    #model.add(Dense(units=neurons, activation='relu'))
    # model.add(Dropout(0.5))
    #model.add(Dense(units=neurons, activation='relu'))


    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X_train, Y_train, batch_size=64, epochs=30)

    # evaluate the model and output the accuracy
    result_train = model.evaluate(X_train, Y_train)
    result_test = model.evaluate(X_test, Y_test)
       
    print('Train Acc:', result_train[1])
    print('Test Acc:', result_test[1])
    predict = model.predict(X_valid)

    pred_df = pd.DataFrame(data={'Filename': df_file_name, 'Obesity': np.round(predict)[:,1]})
    pred_df['Obesity'] = pred_df['Obesity'].astype('int64')
    #pred_df.to_csv("./pred.csv", index=False)
    print(pred_df)
    print(pred_df['Obesity'].value_counts())
    return pred_df

results = pd.DataFrame(columns = ['Filename', 'Obesity'])
result = train_nn_model()
results['Filename'] = result['Filename']
results['Obesity'] = result['Obesity']

cases = 60
for i in range(cases):
    result = train_nn_model()
    #print(results)
    results['Filename'] = result['Filename']
    results['Obesity'] += result['Obesity']

print(results)
print(results['Obesity'].value_counts().sort_index())

results['Obesity']/= cases+1
results['Obesity']+= 0.1
results['Obesity'] = results['Obesity'].round().astype('int64')
# results['Obesity'] = results['Obesity'].astype('int64')

print(results)
print(results['Obesity'].value_counts().sort_index())

results.to_csv("./pred.csv", index=False)
