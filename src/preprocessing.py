import numpy as np
from PIL import Image


def preprocess_image(image):

    # resize to match training size
    image = image.resize((128, 128))

    # convert to numpy array
    image = np.array(image)

    # normalize
    image = image / 255.0

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    return image