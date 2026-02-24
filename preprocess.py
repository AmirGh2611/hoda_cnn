from dataset import load_hoda

# reading data
x_train, y_train, x_test, y_test = load_hoda()
# standardization
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
# uint8 to float32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# add channel
x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)
