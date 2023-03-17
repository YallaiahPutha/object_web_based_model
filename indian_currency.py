"""
 * Copyright (C) 2018-2023 NSLHUB smartvision@nslhub.com  
 * 
 * This file is part of Vision for Visually Impaired (VFVI) Project.
 * 
 * VFVI can not be copied and/or distributed without the express
 * permission of NSLHUB
 """

'''
Summary : For this file we called the Indian currency model and got output as class_name.
'''

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import time
import cv2

fig = plt.figure()
model = "./model/model_indian_adam.h5"

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Currency Classifier')


def indian_currency_classify(img, model):
    """
    We are passing  image_path and model_path as input and got output as bounding_box coordinates and label index positions.
    Args:
        img (path): image file path
        model (path): model file path

    Returns:
        Output (list of coordinates): got output as label index position
    """
    img_width, img_height = 224, 224
    img = np.asarray(img)
    dm = (img_width, img_height)
    img = cv2.resize(img, dm)
    img = img.reshape(1, img_width, img_height, 3)
    print(img)
    img = img/255.0
    model = tf.keras.models.load_model(
            model,
            custom_objects={'KerasLayer': hub.KerasLayer}
    )
    classes = ["10", "100", "20", "200", "2000", "50", "500"]
    print(classes)
    predictions = model(img)
    pre = model.predict(img)
    print("____________________________", pre)
    print("44444$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", predictions)
    pred = tf.nn.softmax(predictions, axis=None, name=None)
    print("##############################", pred)
    result = np.argmax(pred[0])
    if pred[0][result] > 90:
        print(result)
        return classes[result]
    else:
        return 'Non prediction'


def indian_main():
    """
    This function we are calling the detection function and getting the user interface to upload the image bar available.
    """
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        print("In Main function", image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = indian_currency_classify(image, model)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


# if __name__ == "__main__":
#     indian_main()
