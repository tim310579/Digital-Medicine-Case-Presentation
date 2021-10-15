# Case Presentation1

## environment requirement
`$ pip install -r requirements.txt`

## produce train/test/valid data
`$ python preprocess.py`

We use tf-idf algorithm to produce a csv file, which contains the top 50 common words in all medical record.

### train textual csv
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

## run our nn_model
`$ python nn_model.py`
