from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os

# Klasör yolunu belirleyin
klasor_adi = "uploads"
klasor_yolu = os.path.join(os.getcwd(), klasor_adi)

# Klasördeki dosyaları listele
dosyalar = os.listdir(klasor_yolu)

# Klasördeki resim dosyalarını bulun
resimler = [dosya for dosya in dosyalar if dosya.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Resim dosyalarını zaman damgasına göre sırala (en yeni önce gelecek şekilde)
resimler.sort(key=lambda x: os.path.getctime(os.path.join(klasor_yolu, x)), reverse=True)

if resimler:
    # En yeni resmi açın ve yolunu bir değişkene kaydedin
    en_yeni_resim_yolu = os.path.join(klasor_yolu, resimler[0])
    print("En yeni resmin yolu:", en_yeni_resim_yolu)
else:
    print("Klasörde resim dosyası bulunamadı.")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open(en_yeni_resim_yolu).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)

