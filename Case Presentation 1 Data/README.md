# Case Presentation1

## environment requirement
`$ pip install -r requirements.txt`

## produce train/test/valid data(csv)
`$ python preprocess.py`

#### Use tf-idf to selct common words in Y & U record, then select those words which are in Y but not in U(Y-U, which is important in Y)

#### Use the above words to produce a csv file which contains frequency count in every record.

#### Use words produced by tf-idf to generate word count csv file with every record.

### train_tfidf_data.csv

| dilat | aortic | admiss | coronari | bypass | pressur | graft | ... |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.058066243 | 0.215111958 | 0.203194192 | 0 | 0.207217001 | ... |
| 0 | 0.065653761 | 0.043154684 | 0.039967665 | 0 | 0 | 0 | ... |
| 0 | 0 | 0.072424013 | 0.022358474 | 0 | 0.019764213 | 0 | ... |
| 0 | 0 | 0.015108373 | 0.013992603 | 0 | 0.074214222 | 0 | ... |
| 0 | 0 | 0.036473631 | 0 | 0 | 0.044790763 | 0 | ... |
| 0 | 0 | 0.02816331 | 0.052166839 | 0 | 0 | 0 | ... |

## use train_tfidf_data.csv to train nn_model

`$ python nn_model.py`

Train Acc: 0.8224999904632568
Test Acc: 0.8525000214576721
Train f1 score: 0.8065395095367848
Test f1 score: 0.8459530026109661

| Filename | Obesity |
|---|---|
| ID_1159.txt | 0 |
| ID_1160.txt | 1 |
| ID_1162.txt | 1 |
| ...| ... |
| ID_1240.txt | 1 |
| ID_1242.txt | 1 |
| ID_1243.txt | 1 |

| Obesity | Count |
|---|---|
| 0 | 30 |
| 1 | 20 |

