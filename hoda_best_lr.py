from keras.losses import SparseCategoricalCrossentropy
from learningratefinder import LearningRateFinder
from cnn_model import model
from preprocess import x_train, y_train

model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
lr_finder = LearningRateFinder(model)
lr_finder.find((x_train, y_train), 1e-10, 1e+1)
lr_finder.plot_loss()
