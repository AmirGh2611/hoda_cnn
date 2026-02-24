from preprocess import x_test, y_test
from keras.models import load_model

model = load_model("bestmodel_hoda.model.keras")
loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.2f}")
print(f"test accuracy: {acc:.2f}")
