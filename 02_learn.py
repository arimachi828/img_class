import keras
from keras import layers, models
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt


#モデルの作成
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(250,250,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation = "relu"))
model.add(layers.Dense(4,activation = "sigmoid"))

model.summary()


#モデルのコンパイル
model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])




#データ読み込み
categories = ["プライベート","プロフィール","試合時","集合写真"]
nb_classes=len(categories)

X_train,X_test,Y_train,Y_test = np.load("/home/arimachi828/HDD/work2/DL_env2/Image-recognition/glp_data.npy")
X_train = X_train.astype("float")/255
X_test = X_test.astype("float")/255

Y_train = np_utils.to_categorical(Y_train,nb_classes)
Y_test = np_utils.to_categorical(Y_test,nb_classes)

#モデルの学習
model = model.fit(X_train,
                 Y_train,
                 epochs=10,
                 verbose=1,
                 batch_size=6,
                 validation_data=(X_test,Y_test))






#学習結果のグラフ化
acc = model.history['acc']
val_acc=model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('精度を示すグラフのファイル名')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title('Training and validation loss' )
plt.legend()
plt.savefig('損失値を示すグラフのファイル名')




#モデルの保存----------------------------------------------------------------------------------------------
json_string = model.model.to_json()
open('/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_predict.json', 'w').write(json_string)

#重みの保存
hdf5_file = "/home/arimachi828/HDD/work2/DL_env2/image_recog/smile_predict.hdf5"
model.model.save_weights(hdf5_file)
#----------------------------------------------------------------------------------------------------------
