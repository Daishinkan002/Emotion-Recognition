import numpy as np
import matplotlib.image as img

def preprocess_input(datatype, v2=True):
    x = datatype.astype('float32')
    x /= 255.0
    if v2:
        x = (2.0*x) - 1.0
    return x

def read(image_name):
    return img.imread(image_name)

def resize(image_data,size):
    return np.array(Image.fromarray(image_data).resize(size))
