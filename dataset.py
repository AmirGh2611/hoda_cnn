# for more information read "19-Intro2ML-HodaDataset.ipynb"
import cv2
import numpy as np
from scipy import io


def load_hoda(training_sample_size=4000, test_sample_size=800, size=32):
    # load dataset
    trs = training_sample_size
    tes = test_sample_size
    dataset = io.loadmat("data_hoda_full.mat")

    # test and training set
    x_train_orginal = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    x_test_original = np.squeeze(dataset['Data'][trs:trs + tes])
    y_test = np.squeeze(dataset['labels'][trs:trs + tes])

    # resize
    x_train_32by32 = [cv2.resize(img, dsize=(size, size)) for img in x_train_orginal]
    x_test_32by_32 = [cv2.resize(img, dsize=(size, size)) for img in x_test_original]
    # reshape
    x_train = np.reshape(x_train_32by32, (-1, size ** 2))
    x_test = np.reshape(x_test_32by_32, (-1, size ** 2))

    return x_train, y_train, x_test, y_test
