import cv2
from PIL import Image
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from os import listdir
import os

SIZE=512

def to_fuzzy(path, new_path):

    files = listdir(path)
    for file in files:
        print(new_path + file, end=' ')
        if os.path.isfile(new_path + file):
            print('existed')
            continue
        print('generating')
        n = 2 # number of rows (windows on columns)
        m = 2 # number of colomns (windows on rows)
        EPSILON = 0.00001
        #GAMMA, IDEAL_VARIANCE 'maybe' have to changed from image to another
        GAMMA = 1 # Big GAMMA >> Big mean >> More Brightness
        IDEAL_VARIANCE = 0.35 #Big value >> Big variance >> Big lamda >> more contrast

        #img_name = 'swap_data/train/0b643f69a774.jpg'
        img = cv2.imread(path+file)
        #img = cv.resize(img, (200, 200))
        layer = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        WIDTH = layer.shape[1]
        HEIGHT = layer.shape[0]
        x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1


        # split the image to windows
        def phy(value):  # phy: E --> R
            # if ((1+value)/((1-value)+0.0001)) < 0:
            # print(value)
            return 0.5 * np.log((1 + value) / ((1 - value) + EPSILON))


        def multiplication(value1, value2):  # ExE --> R
            return phy(value1) * phy(value2)


        def norm(value):
            return abs(phy(value))


        def scalar_multiplication(scalar, value):  # value in E ([-1,1])
            s = (1 + value) ** scalar
            z = (1 - value) ** scalar
            res = (s - z) / (s + z + EPSILON)
            return res


        def addition(value1, value2):  # value1,value2 are in E ([-1,1])
            res = (value1 + value2) / (1 + (value1 * value2) + EPSILON)
            return res


        def subtract(value1, value2):  # value1,value2 are in E ([-1,1])
            res = (value1 - value2) / (1 - (value1 * value2) + EPSILON)
            return res


        def C(m, i):
            return math.factorial(m) / ((math.factorial(i) * math.factorial(m - i)) + EPSILON)


        def qx(i, x):  # i: window index in rows, x: number of current pixel on x-axis
            if (x == WIDTH - 1):
                return 0
            return C(m, i) * (np.power((x - x0) / (x1 - x), i) * np.power((x1 - x) / (x1 - x0),
                                                                          m))  # This is the seconf implementation
            # return C(m,i)*((np.power(x-x0,i) * np.power(x1-x,m-i)) / (np.power(x1-x0,m)+EPSILON))


        def qy(j, y):
            '''
            The second implementation for the formula does not go into overflow.
            '''
            if (y == HEIGHT - 1):
                return 0
            return C(n, j) * (np.power((y - y0) / (y1 - y), j) * np.power((y1 - y) / (y1 - y0),
                                                                          n))  # This is the seconf implementation
            # return C(n,j)*((np.power((y-y0),j) * np.power((y1-y),n-j))/ (np.power(y1-y0,n)+EPSILON))


        def p(i, j, x, y):
            return qx(i, x) * qy(j, y)


        def mapping(img, source, dest):
            return (dest[1] - dest[0]) * ((img - source[0]) / (source[1] - source[0])) + dest[0]

        e_layer_gray = mapping(layer, (0, 255), (-1, 1))

        def cal_ps_ws(m, n, w, h, gamma):
            ps = np.zeros((m, n, w, h))
            for i in range(m):
                for j in range(n):
                    for k in range(w):
                        for l in range(h):
                            ps[i, j, k, l] = p(i, j, k, l)

            ws = np.zeros((m, n, w, h))
            for i in range(m):
                for j in range(n):
                    ps_power_gamma = np.power(ps[i, j], gamma)
                    for k in range(w):
                        for l in range(h):
                            ws[i, j, k, l] = ps_power_gamma[k, l] / (np.sum(ps[:, :, k, l])+EPSILON)
            return ps, ws
        #print('Ps and Ws calculation is in progress...')
        start = time.time()
        ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
        end = time.time()
        #print('Ps and Ws calculation has completed successfully in '+str(end-start)+' s')


        def cal_means_variances_lamdas(w, e_layer):
            means = np.zeros((m, n))
            variances = np.zeros((m, n))
            lamdas = np.zeros((m, n))
            taos = np.zeros((m, n))

            def window_card(w):
                return np.sum(w)

            def window_mean(w, i, j):
                mean = 0
                for k in range(HEIGHT):
                    for l in range(WIDTH):
                        mean = addition(mean, scalar_multiplication(w[i, j, l, k], e_layer[k, l]))
                mean /= window_card(w[i, j])
                return mean

            def window_variance(w, i, j):
                variance = 0
                for k in range(HEIGHT):
                    for l in range(WIDTH):
                        variance += w[i, j, l, k] * np.power(norm(subtract(e_layer[k, l], means[i, j])), 2)
                variance /= window_card(w[i, j])
                return variance

            def window_lamda(w, i, j):
                return np.sqrt(IDEAL_VARIANCE) / (np.sqrt(variances[i, j]) + EPSILON)

            def window_tao(w, i, j):
                return window_mean(w, i, j)

            for i in range(m):
                for j in range(n):
                    means[i, j] = window_mean(ws, i, j)
                    variances[i, j] = window_variance(ws, i, j)
                    lamdas[i, j] = window_lamda(ws, i, j)
            taos = means.copy()

            return means, variances, lamdas, taos


        #print('means, variances, lamdas and taos calculation is in progress...')
        start = time.time()
        means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_gray)
        end = time.time()
        #print('means, variances, lamdas and taos calculation is finished in ' + str(end - start) + ' s')

        def window_enh(w, i, j, e_layer):
            return scalar_multiplication(lamdas[i, j], subtract(e_layer, taos[i, j]))

        def image_enh(w, e_layer):
            new_image = np.zeros(e_layer.shape)
            width = e_layer.shape[1]
            height = e_layer.shape[0]
            for i in range(m):
                for j in range(n):
                    win = window_enh(w, i, j, e_layer)
                    w1 = w[i, j].T.copy()
                    for k in range(width):
                        for l in range(height):
                            new_image[l, k] = addition(new_image[l, k], scalar_multiplication(w1[l, k], win[l, k]))
            return new_image

        def one_layer_enhacement(e_layer):
            #card_image = layer.shape[0]*layer.shape[1]
            new_E_image = image_enh(ws, e_layer)
            res_image = mapping(new_E_image, (-1, 1), (0, 255))
            res_image = np.round(res_image)
            res_image = res_image.astype(np.uint8)
            return res_image

        res_img = one_layer_enhacement(e_layer_gray)

        #new_path = 'to_fuzzy/train/'
        cv2.imwrite(new_path+file, res_img)
        #res_img.save(new_path+file)
        #stop

def method1(path):
    img_gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return img_bin

def img_stacking(path1, path2, new_path):
    files1 = files = listdir(path1)
    files2 = files = listdir(path2)
    files1.sort()
    files2.sort()
    opacity = 0.6
    contrast = 1.5
    brightness = -80


    for file1, file2 in zip(files1, files2):
        if file1 != file2: print('fkk')
        img1 = cv2.imread(path1 + file1)
        img2 = cv2.imread(path2 + file2)

        #res_img = cv2.add(img1, img2)
        img = cv2.addWeighted( img1, contrast, img2, opacity, brightness)
        #img = img1 * contrast + img2*opacity + brightness
        #img = cv2.convertScaleAbs(res_img, alpha=contrast, beta=brightness)

        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        cv2.imwrite(new_path + file1, img)

        x = np.array(img)
        x = x[:,:,2]
        x = x / 255

        pix1, pix2, pix3, pix4 = 0, 0, 0, 0  # 左上，右上，左下，右下
        for i in range(int(SIZE / 8)):
            for j in range(int(SIZE / 4)):
                pix1 += x[i, j]  # 左上
                pix2 += x[i, SIZE-1 - j]  # 右上
                pix3 += x[SIZE-1 - j, i]  # 左下
                pix4 += x[SIZE-1 - j, SIZE-1 - i]  # 右下
        #print(pix2)
        if pix1 + pix2 <= 64 * 128 * 2 / 3:  # 上面四個位置偏黑
            x = 1 - x
            #print(file1)
            #cnt += 1
            # print(im, pix1+pix2)
        elif pix3 + pix4 <= 64 * 128 * 2 / 3:  # 左右下偏黑
            x = 1 - x
            #print(file1)

        x = 1 - x

        img2 = Image.fromarray(x * 255)  # numpy轉image类
        # img2.show()

        # print(new_fullpath)
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        img2.save(new_path + file1)
        #cv2.imwrite(new_path + file1, img2)

if __name__ == '__main__':
    to_fuzzy('resized_data/train/', 'to_fuzzy/train/')
    to_fuzzy('resized_data/valid/', 'to_fuzzy/valid/')

    img_stacking('resized_data/train/', 'to_fuzzy/train/', 'stacking_data/train/')
    img_stacking('resized_data/valid/', 'to_fuzzy/valid/', 'stacking_data/valid/')
