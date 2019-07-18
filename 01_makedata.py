from PIL import Image
import os,  glob
import numpy as np
import random, math

#画像が保存されているデータセットフォルダ
root_dir = "/home/arimachi828/HDD/work2/DL_env2/image_recog/datasets10000"
#分類したいカテゴリ
categories = ["オフショット","プロフィール","試合","集合写真"]

#画像データ用配列
X = []
#ラベルデータ用配列
Y = []


def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)


def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((150,150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)


allfiles = []

for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))


random.shuffle(allfiles)
th = math.floor(len(allfiles)*0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train,Y_train = make_sample(train)
X_test,Y_test = make_sample(test)
xy = (X_train,X_test,Y_train,Y_test)
#データを保存する
np.save("/home/arimachi828/HDD/work2/DL_env2/image_recog/glp_data.npy",xy)
