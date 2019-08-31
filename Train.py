import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from CNNModel.Training.Generate import ImageInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
from CustomerPath import train_folder, validation_folder, store_folder

input_shape = [496, 496, 3]
batch_size = 4

if not os.path.exists(store_folder):
    os.mkdir(store_folder)

number_training = len(os.listdir(train_folder))
number_validation = len(os.listdir(validation_folder))

# Generate
train_generator = ImageInImageOut2D(train_folder, input_shape[:1], batch_size=batch_size, augment_param=random_2d_augment)
validation_generator = ImageInImageOut2D(validation_folder, input_shape[:1], batch_size=batch_size, augment_param=random_2d_augment)

# Model
from UNet import UNet
from CNNModel.Utility.SaveAndLoad import SaveModel

model = UNet(input_shape, channel=6)
SaveModel(model, store_folder)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
    EarlyStopping(monitor='val_loss', patience=100, mode='min'),
    ModelCheckpoint(filepath=os.path.join(store_folder, 'best_weights.h5'), monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
]

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=number_training // batch_size, epochs=1000, verbose=1,
                              validation_data=validation_generator, validation_steps=number_validation // batch_size,
                              callbacks=callbacks)

model.save_weights(os.path.join(store_folder, 'last_weights.h5'))

from visualization import show_train_history
show_train_history(history, 'loss', 'val_loss')







