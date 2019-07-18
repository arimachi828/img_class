#未知のデータで精度を測る-------------------------------------------------------------------------------------
from PIL import Image
import os, glob
import numpy as np
import random, math

# 画像が保存されているディレクトリのパス
root_dir = "/home/arimachi828/HDD/work2/DL_env2/image_recog/datasets_test"
# 画像が保存されているフォルダ名
categories = ["プライベート","プロフィール","試合時","集合写真"]

X = [] # 画像データ
Y = [] # ラベルデータ

allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

for cat, fname in allfiles:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

x = np.array(X)
y = np.array(Y)
#データをnpy形式で保存
np.save("/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_data_test_X_150.npy", x)
np.save("/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_data_test_Y_150.npy", y)



#データの呼び出し
test_X = np.load("smile_data_test_X_150.npy")
test_Y = np.load("smile_data_test_Y_150.npy")

from keras.utils import np_utils
test_Y = np_utils.to_categorical(test_Y,4)

score = model.model.evaluate(x=test_X,y=test_Y)

print('loss=',score[0])
print('accuracy',score[1])
#-------------------------------------------------------------------------------------------------------------
