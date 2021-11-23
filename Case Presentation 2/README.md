# Case Presentation2

* 原始data檔案太大，就不放上來了，只放轉換成圖片的原始大小圖片(to_image_data)

## Data preprocessing 

* ### environment requirement
* #### `$ pip install -r requirements.txt`

* ### resize origin image to the same size
* `$ python preprocess.py`

* ### generate fuzzy & stacking image data
* `$ python fuzzy.py`

## Result
* ### Train model and generate result
* `$ python model.py`


* 若電腦沒有GPU，建議直接使用kaggle notebook執行程式

* notebook連結
