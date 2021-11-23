import SimpleITK as sitk
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import cv2
import numpy as np
import pandas as pd
import time, math

#im = Image.open('00af6f8c2a3d.jpg')
SIZE = 512

def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 歸一化
    newimg = (newimg * 255).astype('uint8')  # 將畫素值擴充套件到[0,255]
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def save_jpg(dir_path, path, filename):
    #print(path, filename)
    dcm_image_path = path  # 讀取dicom檔案
    output_jpg_path = join(dir_path, filename.replace('dcm', 'jpg'))
    ds_array = sitk.ReadImage(dcm_image_path)  # 讀取dicom檔案的相關資訊
    img_array = sitk.GetArrayFromImage(ds_array)  # 獲取array

    shape = img_array.shape
    img_array = np.reshape(img_array, (shape[1], shape[2]))  # 獲取array中的height和width
    high = np.max(img_array)
    low = np.min(img_array)
    convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)

def resize_picture(path, new_path):
    files = listdir(path)
    files.sort()

    for every_image in files:
        fullpath = join(path, every_image)
        if '.jpg' in every_image:
            # print('haha')
            im = Image.open(fullpath)

            # print(fullpath)
            new_img = im.resize((SIZE, SIZE), Image.BILINEAR)  # change to the same size
            #img2 = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
            #img2 = img2[int(size / 8): int(size * 7 / 8), int(size / 8): int(size * 7 / 8)]  # 邊邊裁掉
            #resize_image = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            new_fullpath = join(new_path, every_image)
            new_img.save(new_fullpath)


def pull_from_multi_dir(path, new_path):
    files = listdir(path)
    all_dcm_filename = []
    all_dcm_path = []
    for file in files:
        dcm_dir = listdir(join(path, file))[0]
        all_dcm_filename.append(listdir(join(path, file, dcm_dir))[0])
        all_dcm_path.append(join(path, file, dcm_dir, all_dcm_filename[-1]))

    for i in range(len(all_dcm_path)):
        save_jpg(new_path, all_dcm_path[i], all_dcm_filename[i])
        

if __name__ == '__main__':

    path1 = 'data/train'
    path2 = 'data/valid'

    new_path1 = 'to_image_data/train/'
    new_path2 = 'to_image_data/valid/'

    #pull_from_multi_dir(path1, new_path1)
    #print('train dcm->image')
    #pull_from_multi_dir(path2, new_path2)
    #print('valid dcm->image')

    resize_picture(new_path1, 'resized_data/train/')
    resize_picture(new_path2, 'resized_data/valid')
    #print('resize to 512 ok')

