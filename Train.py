import os

import matplotlib.pyplot as plt

from Generate import ImageInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
<<<<<<< Updated upstream
import os
from CNNModel.Training.Generate import ImageInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
from CustomerPath import train_folder, validation_folder, store_folder

input_shape = [496, 496, 3]
=======
from LossFunction import dice_coef_loss


train_folder = r'D:\Gleason2019\TrainValidationTest_256\TrainValidationTest_256\Train'
validation_folder = r'D:\Gleason2019\TrainValidationTest_256\TrainValidationTest_256\Test'
store_folder = r'D:\Gleason2019\TrainValidationTest_256\model_categorical_crossentropy'
input_shape = [240, 240, 3]
>>>>>>> Stashed changes
batch_size = 4

if not os.path.exists(store_folder):
    os.mkdir(store_folder)

number_training = len(os.listdir(train_folder))
number_validation = len(os.listdir(validation_folder))

# Generate
<<<<<<< Updated upstream
train_generator = ImageInImageOut2D(train_folder, (496, 496), batch_size=batch_size, augment_param=random_2d_augment)
validation_generator = ImageInImageOut2D(validation_folder, (496, 496), batch_size=batch_size, augment_param=random_2d_augment)
=======
train_generator = ImageInImageOut2D(train_folder, (240, 240), batch_size=batch_size, augment_param=random_2d_augment)
validation_generator = ImageInImageOut2D(validation_folder, (240, 240), batch_size=batch_size, augment_param=random_2d_augment)
>>>>>>> Stashed changes

# Model
from UNet import UNet
<<<<<<< Updated upstream
from saveandload import SaveModel
=======
from MeDIT.CNNModel.SaveAndLoad import SaveModel,SaveHistory



import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()

config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

sess = tf.Session(config=config)

KTF.set_session(sess)

>>>>>>> Stashed changes

model = UNet(input_shape, channel=6)
SaveModel(model, store_folder)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
    EarlyStopping(monitor='val_loss', patience=100, mode='min'),
    ModelCheckpoint(filepath=os.path.join(store_folder, 'best_weights.h5'), monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
]


<<<<<<< Updated upstream
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=number_training // batch_size, epochs=1000, verbose=1,
                              validation_data=validation_generator, validation_steps=number_validation // batch_size,
                              callbacks=callbacks)
=======
# model.compile(loss=dice_coef_loss, optimizer=Adam(0.001), metrics=[dice_coef_loss])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# history = model.fit_generator(train_generator, steps_per_epoch=number_training // batch_size, epochs=1000, verbose=1,
#                               validation_data=validation_generator, validation_steps=number_validation // batch_size,
#                               callbacks=callbacks)
# SaveHistory(history.history, store_folder)
#
# model.save_weights(os.path.join(store_folder, 'last_weights.h5'))
import pickle

f = open(os.path.join(store_folder, 'history_dict.txt'), 'rb')
history = pickle.load(f)
f.close()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['train', 'val'])
plt.title('Loss history')
plt.savefig(os.path.join(store_folder, 'loss_curve.png'))
plt.show()


>>>>>>> Stashed changes

# from visualization import show_train_history
# show_train_history(history, 'loss', 'val_loss')







