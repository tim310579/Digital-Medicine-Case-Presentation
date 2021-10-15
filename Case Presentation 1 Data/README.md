# Case Presentation1

## environment requirement
`$ pip install -r requirements.txt`

## produce train/test/valid data(csv)
`$ python preprocess.py`

We use tf-idf algorithm to produce a csv file, which contains the top 50 common words in all medical record.

### tf-idf result

| | cornory | chronic | diabetes | gout | hypertension | ... |
|---|---|---|---|---|---|---|
| count | 400 | 400 | 400 | 400 | 400 | ... | 
| mean | 0.022213 | 0.012447 | 0.017089 | 0.005995 | 0.013830 | ... |
| std |
| min |
| 25% |
| 50% |
| 75% |
| max |

### train_data.csv

| id | is_Obese| cornory | chronic | diabetes | gout | hypertension | ... |
|---|---|---|---|---|---|---|---|
| 0 | 7 | 0 | 0 | 0 | 0 | ... |
| 1 | 0 | 1 | 0 | 0 | 0 | ... |
| 2 | 0 | 2 | 0 | 2 | 0 | ... |
| 3 | 0 |	1 |	0	| 3	| 0 | ... |
| 4 | 0 |	0 |	0	| 3	| 0 | ... |
| 5 | 0 | 2 | 0 | 2 | 0 | ... |

## run train_data.csv to train nn_model
`$ python nn_model.py`
