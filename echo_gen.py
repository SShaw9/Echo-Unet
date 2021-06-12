import os
import random
import echo_unet
import echo_metrics as em
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageOps
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.metrics import Recall, Precision

echonet_dir = Path('../Echonet/')
# original_train_dir = echonet_dir / 'Images' / 'Train' / 'EDV' / 'Originals2'
original_train_dir_all = echonet_dir / 'Images' / 'Train' / 'AllOriginals'
# original_val_dir = echonet_dir / 'Images' / 'Val' / 'EDV' / 'Originals2'
original_val_dir_all = echonet_dir / 'Images' / 'Val' / 'AllOriginalsGray'
# mask_train_dir = echonet_dir / 'Images' / 'Train' / 'EDV' / 'Masks2'
mask_train_dir_all = echonet_dir / 'Images' / 'Train' / 'AllMasks'
# mask_val_dir = echonet_dir / 'Images' / 'Val' / 'EDV' / 'Masks2'
mask_val_dir_all = echonet_dir / 'Images' / 'Val' / 'AllMasksGray'

IMG_SIZE = (112, 112)
num_classes = 2
BATCH_SIZE = 64


input_train_original_paths = sorted(
    [
        os.path.join(original_train_dir_all, fname)
        for fname in os.listdir(original_train_dir_all)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

input_train_mask_paths = sorted(
    [
        os.path.join(mask_train_dir_all, fname)
        for fname in os.listdir(mask_train_dir_all)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

val_original_paths = sorted(
    [
        os.path.join(original_val_dir_all, fname)
        for fname in os.listdir(original_val_dir_all)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

val_mask_paths = sorted(
    [
        os.path.join(mask_val_dir_all, fname)
        for fname in os.listdir(mask_val_dir_all)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

print("Number of training inputs ", len(input_train_original_paths))
print("Number of val inputs ", len(val_original_paths))
print("Number of training inputs ", len(input_train_mask_paths))
print("Number of val inputs ", len(val_mask_paths))

for original, mask in zip(input_train_original_paths[:10], input_train_mask_paths[:10]):
    print(original, '|', mask)
for valoriginal, valmask in zip(val_original_paths[:10], val_mask_paths[:10]):
    print(valoriginal, '|',  valmask)

"""
    Main class for tensorflow generation object.
    Uses the keras Sequence object as a base class.
    This obect is responsible for feed the data into the network.
"""
class EchoGen(keras.utils.Sequence):

    def __init__(self, original_paths, mask_paths, batch_size, img_size):
        self.original_paths = original_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.mask_paths) // self.batch_size

    def on_epoch_end(self):
        random.Random(1337).shuffle(self.original_paths)
        random.Random(1337).shuffle(self.mask_paths)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_original_paths = self.original_paths[i: i + self.batch_size]
        batch_mask_paths = self.mask_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_original_paths):
            i_img = cv2.imread(path, cv2.IMREAD_COLOR)
            # normalise
            i_img = cv2.resize(i_img, IMG_SIZE)
            i_img = i_img / 255
            i_img = i_img.astype(np.float32)
            x[j] = i_img
            # plt.imshow(i_img)
            # plt.show()
            # print(i_img)

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='float32')
        for j, path in enumerate(batch_mask_paths):
            m_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            m_img = cv2.resize(m_img, IMG_SIZE)
            # noramalise
            m_img[m_img > 0] = 255
            m_img = m_img / 255
            m_img = m_img.astype(np.int32)
            m_img = np.expand_dims(m_img, 2)
            y[j] = m_img
            # plt.imshow(m_img)
            # plt.show()
            # print(m_img)

        return x, y


# keras.backend.clear_session()
model = echo_unet.build_unet()
model.summary()


train_gen = EchoGen(input_train_original_paths, input_train_mask_paths, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_gen = EchoGen(val_original_paths, val_mask_paths, batch_size=BATCH_SIZE, img_size=IMG_SIZE)


opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
dice_loss = em.dice_loss
dice_metric = em.dice_coef
metrics = [dice_metric, Precision(), Recall()]

model_checkpoint = [
    keras.callbacks.ModelCheckpoint("weights.h5", monitor='val_loss', save_best_only=True)
]

tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)

# Compile Model
model.compile(optimizer=opt, loss=dice_loss, metrics=metrics)
# Fit Model
epochs = 3
model.fit(train_gen,
          epochs=epochs,
          validation_data=val_gen)


val_gen_pred = EchoGen(val_original_paths, val_mask_paths, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_preds = model.predict(val_gen_pred)


def parse_preds(preds):
    """ Convert probabilities into object and background pixel values based on a threshold. """
    _preds = (preds > 0.5).astype(int)
    _preds = _preds * 255
    print(_preds.shape)
    return _preds


def display_pred(i):
    """ Display predictions. """
    preds = parse_preds(val_preds)
    _mask = preds[i]
    print(_mask.shape)
    _mask = array_to_img(_mask)
    plt.imshow(_mask)
    plt.show()


example = 10
# Display input image
img = load_img(val_original_paths[example])
plt.imshow(img)
plt.show()
# Display ground-truth target mask
target = PIL.ImageOps.autocontrast(load_img(val_mask_paths[example]))
plt.imshow(target)
plt.show()
# Display mask predicted by our model
display_pred(example)



