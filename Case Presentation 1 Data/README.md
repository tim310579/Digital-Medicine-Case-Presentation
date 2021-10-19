# Case Presentation1

## environment requirement
`$ pip install -r requirements.txt`

## produce train/test/valid data(csv)
`$ python preprocess.py`

Use tf-idf algorithm to selct common words in Y & U record, then select those words which are in Y but not in U(Y-U), and in U not in Y(U-Y).
Use the above words to produce a csv file which contains frequency count in every record.

### train_tfidf_data.csv

Use words produced by tf-idf to generate word count csv file with every record.

| dilat	| aortic | admiss	| coronari | bypass	| pressur | ... |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 7 | 0 | 0 | 0 | 0 | ... |
| 1 | 0 | 1 | 0 | 0 | 0 | 0 | ... |
| 2 | 0 | 2 | 0 | 2 | 0 | 0 | ... |
| 3 | 0 |	1 |	0	| 3	| 0 | 0 | ... |
| 4 | 0 |	0 |	0	| 3	| 0 | 0 | ... |
| 5 | 0 | 2 | 0 | 2 | 0 | 0 | ... |

## run train_data.csv to train nn_model
`$ python nn_model.py`
