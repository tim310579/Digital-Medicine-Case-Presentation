# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.884650Z","iopub.execute_input":"2021-10-05T07:29:17.884997Z","iopub.status.idle":"2021-10-05T07:29:17.904619Z","shell.execute_reply.started":"2021-10-05T07:29:17.884966Z","shell.execute_reply":"2021-10-05T07:29:17.903702Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.906335Z","iopub.execute_input":"2021-10-05T07:29:17.906868Z","iopub.status.idle":"2021-10-05T07:29:17.920208Z","shell.execute_reply.started":"2021-10-05T07:29:17.906835Z","shell.execute_reply":"2021-10-05T07:29:17.919244Z"}}
del_col = ['is_Obese', 'text_obese']

df = pd.read_csv('train_data.csv')
#print(df)

Y_train = df['is_Obese']
X_train = df
X_train = X_train.drop(columns=del_col)
#print(X_train, Y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.921908Z","iopub.execute_input":"2021-10-05T07:29:17.922422Z","iopub.status.idle":"2021-10-05T07:29:17.935708Z","shell.execute_reply.started":"2021-10-05T07:29:17.922379Z","shell.execute_reply":"2021-10-05T07:29:17.934414Z"}}
df_test = pd.read_csv('test_data.csv')

Y_test = df_test['is_Obese']
X_test = df_test
X_test = X_test.drop(columns=del_col)
#print(X_test, Y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.937646Z","iopub.execute_input":"2021-10-05T07:29:17.938308Z","iopub.status.idle":"2021-10-05T07:29:17.948349Z","shell.execute_reply.started":"2021-10-05T07:29:17.938269Z","shell.execute_reply":"2021-10-05T07:29:17.947347Z"}}
df_valid = pd.read_csv('valid_data.csv')

df_file_name = df_valid['is_Obese']
X_valid = df_valid
X_valid = X_valid.drop(columns=del_col)
#print(file_name)
#print(X_valid)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.950217Z","iopub.execute_input":"2021-10-05T07:29:17.950759Z","iopub.status.idle":"2021-10-05T07:29:17.958355Z","shell.execute_reply.started":"2021-10-05T07:29:17.950723Z","shell.execute_reply":"2021-10-05T07:29:17.957431Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-10-05T07:29:17.959991Z","iopub.execute_input":"2021-10-05T07:29:17.960242Z","iopub.status.idle":"2021-10-05T07:29:18.256567Z","shell.execute_reply.started":"2021-10-05T07:29:17.960215Z","shell.execute_reply":"2021-10-05T07:29:18.255371Z"}}
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#model = DecisionTreeClassifier().fit(X_train, Y_train)
model = RandomForestClassifier(
                        # class_weight={0:1,1:5}
#                       criterion='entropy', max_depth=8, max_features=0.7, max_leaf_nodes=17,
 #                      min_samples_leaf=17, min_samples_split=30,
  #                    min_weight_fraction_leaf=0.01, random_state=123
                        ).fit(X_train, Y_train)
#model = LogisticRegression().fit(X_train, Y_train)

report_model(model, X_train, Y_train, X_test, Y_test, X_valid)
np.set_printoptions(suppress=True)
print('feature_importances_:', model.feature_importances_)
#report_model(model_2, X_train, Y_train, X_test, Y_test, X_valid)
#report_model(model_3, X_train, Y_train, X_test, Y_test, X_valid)

