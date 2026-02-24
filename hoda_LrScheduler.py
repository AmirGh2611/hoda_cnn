from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import TensorBoard
from cnn_model import model
from preprocess import x_train, y_train
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint


def scheduler(epoch):
    initial_lr = 0.001
    drop_rate = 0.5
    epochs_drop = [10, 20]

    if epoch in epochs_drop:
        new_lr = initial_lr * (drop_rate ** (epochs_drop.index(epoch) + 1))
    else:
        drops_passed = sum(1 for e in epochs_drop if e < epoch)
        new_lr = initial_lr * (drop_rate ** drops_passed)

    return new_lr


lr_scheduler = LearningRateScheduler(scheduler)
tensorboard = TensorBoard(log_dir="logs", histogram_freq=1)
check_point = ModelCheckpoint(filepath="bestmodel_hoda.model.keras", save_best_only=True, monitor='val_accuracy', )
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2,
          callbacks=[tensorboard, lr_scheduler, check_point])
