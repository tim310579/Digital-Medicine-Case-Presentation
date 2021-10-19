# Case Presentation1

## environment requirement
`$ pip install -r requirements.txt`

## produce train/test/valid data(csv)
`$ python preprocess.py`

#### Use tf-idf to selct common words in Y & U record, then select those words which are in Y but not in U(Y-U), and in U not in Y(U-Y).

#### Use the above words to produce a csv file which contains frequency count in every record.

#### Use words produced by tf-idf to generate word count csv file with every record.

### train_tfidf_data.csv

| dilat	| aortic | admiss	| coronari | bypass	| pressur | graft | ... |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.058066243 | 0.215111958 | 0.203194192 | 0 | 0.207217001 | ... |
| 0 | 0.065653761 | 0.043154684 | 0.039967665 | 0 | 0 | 0 | ... |
| 0 | 0 | 0.072424013 | 0.022358474 | 0 | 0.019764213 | 0 | ... |
| 0 | 0 | 0.015108373 | 0.013992603 | 0 | 0.074214222 | 0 | ... |
| 0 | 0 | 0.036473631 | 0 | 0 | 0.044790763 | 0 | ... |
| 0 | 0 | 0.02816331 | 0.052166839 | 0 | 0 | 0 | ... |

## use train_tfidf_data.csv to train nn_model

`$ python nn_model.py`

Train Acc: 0.7857142686843872
Test Acc: 0.6200000047683716
       Filename  Obesity
0   ID_1159.txt        0
1   ID_1160.txt        0
2   ID_1162.txt        0
3   ID_1167.txt        0
4   ID_1168.txt        1
5   ID_1176.txt        0
6   ID_1180.txt        0
7   ID_1183.txt        0
8   ID_1184.txt        1
9   ID_1185.txt        1
10  ID_1186.txt        1
11  ID_1187.txt        0
12  ID_1189.txt        1
13  ID_1190.txt        0
14  ID_1191.txt        0
15  ID_1193.txt        1
16  ID_1194.txt        0
17  ID_1197.txt        1
18  ID_1198.txt        1
19  ID_1200.txt        0
20  ID_1201.txt        1
21  ID_1202.txt        0
22  ID_1203.txt        0
23  ID_1205.txt        0
24  ID_1208.txt        0
25  ID_1209.txt        0
26  ID_1210.txt        1
27  ID_1211.txt        1
28  ID_1212.txt        0
29  ID_1213.txt        1
30  ID_1214.txt        1
31  ID_1216.txt        0
32  ID_1217.txt        0
33  ID_1219.txt        0
34  ID_1220.txt        1
35  ID_1222.txt        0
36  ID_1223.txt        1
37  ID_1226.txt        0
38  ID_1229.txt        0
39  ID_1232.txt        1
40  ID_1233.txt        0
41  ID_1234.txt        0
42  ID_1235.txt        0
43  ID_1236.txt        0
44  ID_1237.txt        1
45  ID_1238.txt        1
46  ID_1239.txt        0
47  ID_1240.txt        1
48  ID_1242.txt        0
49  ID_1243.txt        1
0    30
1    20

