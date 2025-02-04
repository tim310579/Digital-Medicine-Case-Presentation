# Case Presentation2

* 原始data檔案太大，就不放上來了，只放轉換成圖片的原始大小圖片(to_image_data)
* github website的檔案列表最多只會顯示1000筆檔案，但1200筆都存在repository，實際clone時不會出問題
* to_image_data : 原始檔案轉成圖片
* resized_data : 統一圖片大小為512 x 512
* to_fuzzy : 做完fuzzy technique後的圖片
* stacking_data : 將resize後的圖片與fuzzy圖片做疊加(也是最後採用的data)

## Data preprocessing 

### environment requirement
* `$ pip install -r requirements.txt`

### origin data
<img src="https://github.com/tim310579/Digital-Medicine-Case-Presentation/blob/main/Case%20Presentation%202/to_image_data/train/00af6f8c2a3d.jpg" width="20%">

### resize origin image to the same size
* `$ python preprocess.py`

<img src="https://github.com/tim310579/Digital-Medicine-Case-Presentation/blob/main/Case%20Presentation%202/resized_data/train/00af6f8c2a3d.jpg" width="20%">

### generate fuzzy & stacking image data
* `$ python fuzzy.py`

<img src="https://github.com/tim310579/Digital-Medicine-Case-Presentation/blob/main/Case%20Presentation%202/stacking_data/train/00af6f8c2a3d.jpg" width="20%">

## Result
### Train model and generate result

* `$ jupyter notebook`

* 然後直接執行case2-model.ipynb

* 執行結果(submit.csv)，也可參考 [case2-model.ipynb](https://github.com/tim310579/Digital-Medicine-Case-Presentation/blob/main/Case%20Presentation%202/case2-model.ipynb)

| FileID | Type |
|---|---|
|014cc6362544|Atypical|
|014f6b975233|Typical|
|...|...|
|fd0516801814|Typical|

* 若電腦沒有GPU，建議直接使用kaggle notebook執行程式，也可省去安裝環境的麻煩

* notebook連結: [點我](https://www.kaggle.com/tim310579/case2-model)
