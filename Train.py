from data_augmentation.augmentation import AugmentTrain
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from loss_function import dice_coef_loss
import os

train_folder = r'H:\data\data\train'
validation_folder = r'H:\data\data\validation'
store_folder = r'H:/data/TZ roi/savemodel'
input_shape = [200, 200, 1]
batch_size = 16

if not os.path.exists(store_folder):
    os.mkdir(store_folder)

number_training = len(os.listdir(train_folder))
number_validation = len(os.listdir(validation_folder))

# Generate
train_generator = AugmentTrain(train_folder, batch_size)
validation_generator = AugmentTrain(validation_folder, batch_size)


# Model
from u_net import UNet
from saveandload import SaveModel

model = UNet(input_shape)
SaveModel(model, store_folder)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
    EarlyStopping(monitor='val_loss', patience=100, mode='min'),
    ModelCheckpoint(filepath=os.path.join(store_folder, 'best_weights.h5'), monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
]


model.compile(loss=dice_coef_loss, optimizer=Adam(0.001), metrics=[dice_coef_loss])

history = model.fit_generator(train_generator, steps_per_epoch=number_training // batch_size, epochs=1000, verbose=1,
                              validation_data=validation_generator, validation_steps=number_validation // batch_size,
                              callbacks=callbacks)

model.save_weights(os.path.join(store_folder, 'last_weights.h5'))
from visualization import show_train_history
show_train_history(history, 'loss', 'val_loss')








