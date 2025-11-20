import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('digit_classifier.keras')

model.load_weights('digit_classifier.weights.h5')

prediction_image = Image.open('"C:\Users\abrah\OneDrive\Desktop\workshop\hand.png"').convert('L')  # Grayscale
prediction_image = prediction_image.resize((28, 28))  # standard input size for the MNIST dataset that this type of digit classifier model is trained on.

prediction_image = np.array(prediction_image)
prediction_image = prediction_image.reshape(1, 28, 28, 1).astype('float32') / 255.0 #predicting one image at a time,width,height,channelone gray

predictions = model.predict(prediction_image)
print(predictions)
predicted_class = np.argmax(predictions)

print(f"The predicted digit is: {predicted_class}")
