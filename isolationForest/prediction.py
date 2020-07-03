from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest
import pickle
import cv2
import numpy as np
import sklearn

def get_result(path):
    autoencoder = load_model('/home/jbmai/try/modelServer/isolationForest/auto_encoder.h5')
    mms = pickle.load(open('/home/jbmai/try/modelServer/isolationForest/mms.pkl', 'rb'))
    forest = pickle.load(open('/home/jbmai/try/modelServer/isolationForest/forest.pkl', 'rb'))

    img = cv2.imread(path)
    print(path)
    img = cv2.resize(img, (336, 336))
    test = np.array([img])
    # test = test + 20
    test = test/255
    test = test.astype('float32')
    # print(np.max(test))

    features = autoencoder.predict(test)
    features = np.reshape(features, (features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]))
    scaled_features = mms.transform(features)
    result = forest.predict(scaled_features)
    return(result)
