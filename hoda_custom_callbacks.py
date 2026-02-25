from keras.losses import SparseCategoricalCrossentropy
from preprocess import x_train, y_train
from cnn_model import model
from keras.callbacks import Callback


class CustomCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        acc = logs.get("accuracy")
        if acc > 0.99:
            self.model.stop_training = True
            print("\ntrain stopped!")


my_callback = CustomCallback()
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, callbacks=[my_callback])
