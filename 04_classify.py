#画像分類の予測を出力するプログラム-----------------------------------------------------------------------------
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_predict.json').read())
#保存した重みの読み込み
model.load_weights("/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_predict.hdf5")

categories = ["オフショット","プロフィール","試合","集合写真"]


#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

#予測結果によって処理を分ける
if features[0,0] == 1:
    print ("楽しそうなオフショット写真ですね")

elif features[0,1] == 1:
    print ("真剣なプロフィール写真ですね")

else:
    for i in range(0,4):
          if features[0,i] == 1:
              cat = categories[i]
    message = "あなたが選んでいるのは「" + cat + "」の写真ではありませんか？"
    print(message)
#----------------------------------------------------------------------------------------------------------    
