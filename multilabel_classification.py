"""
 * Copyright (C) 2018-2023 NSLHUB smartvision@nslhub.com  
 * 
 * This file is part of Vision for Visually Impaired (VFVI) Project.
 * 
 * VFVI can not be copied and/or distributed without the express
 * permission of NSLHUB
 """

'''
Summary : For this file we called the MultiLabel Classifcation model and got output as multiple class_name.
'''


import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import tensorflow.lite as tflite


fig = plt.figure()
model_path = "./tflite_models/multilabel.tflite"

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('MultiLabel Classifier')


def multilabel_main():
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
                predictions = inference_multilabel(image, model_path)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def inference_multilabel(image_path, model_path):
    """
    We are passing  image_path and model_path as input and got output as bounding_box coordinates and label index positions.
    Args:
        image_path (path): image file path
        modelpath (path): model file path

    Returns:
        Output (list of coordinates): got output as label index position
    """
    classes = [
        "aeroplane",
        "airconditioner",
        "applered",
        "autorikshaw",
        "avocado",
        "banana",
        "bangle",
        "bed",
        "belts",
        "bench",
        "bicycle",
        "binderclips",
        "bittergourd",
        "blanket",
        "book",
        "bookshelf",
        "bottlegourd",
        "brinjal",
        "broadbeans",
        "broccoli",
        "bucket",
        "buffalo",
        "bus",
        "calander",
        "calculator",
        "callingbell",
        "canestick",
        "caps",
        "car",
        "carrot",
        "carrybag",
        "cartonbox",
        "cat",
        "ceilingfan",
        "chair",
        "chalkpeice",
        "chapatistick",
        "cherry",
        "choppingboard",
        "clusterbeans",
        "coconut",
        "colourcrayons",
        "comb",
        "computermouse",
        "cooker",
        "corianderleaves",
        "cow",
        "cremaicbowl",
        "cremaicplates",
        "cucumber",
        "cup",
        "cupboard",
        "curryleaves",
        "curtain",
        "cushion",
        "custardapple",
        "diningtable",
        "directionsignboard",
        "dog",
        "door",
        "doorhandle",
        "drawer",
        "dresses",
        "drumsticks",
        "duster",
        "earphone",
        "earrings",
        "eraser",
        "facemask",
        "fan",
        "file",
        "fiverupeecoin",
        "foodcontainer",
        "fork",
        "formalshoes",
        "fryingpan",
        "garlic",
        "gaslighter",
        "gasstove",
        "gate",
        "gingerroot",
        "glass",
        "glassbottle",
        "glassbowl",
        "glasstable",
        "globe",
        "goat",
        "grapeblue",
        "grapewhite",
        "greenbeans",
        "greencapsicum",
        "greenchili",
        "guava",
        "hairband",
        "hairclip",
        "handbags",
        "handwovencot",
        "heavytruck",
        "helmet",
        "hen",
        "hotbox",
        "iceapple",
        "idol",
        "inductionstove",
        "jamunfruit",
        "jar",
        "jeans",
        "jug",
        "kettle",
        "keyboard",
        "keys",
        "kidneybeans",
        "kitchencabinet",
        "kiwi",
        "knife",
        "ladiesfinger",
        "laptop",
        "leafyvegetables",
        "ledpanellight",
        "lemon",
        "lpgcylinder",
        "lunchbox",
        "mango",
        "manhole",
        "marker",
        "microwaveoven",
        "mixiegrinder",
        "mobilecharger",
        "mobilephone",
        "mosambi",
        "motorcycle",
        "mug",
        "nailcutter",
        "nailpolish",
        "onerupeecoin",
        "onions",
        "orange",
        "papaya",
        "paper",
        "paperclip",
        "paperweight",
        "pen",
        "penstand",
        "pig",
        "pillow",
        "pineapple",
        "plant",
        "pomegranate",
        "pot",
        "potatoes",
        "protractor",
        "pumpkin",
        "radish",
        "railtrack",
        "raspberry",
        "redchilli",
        "refrigerator",
        "remote",
        "ridgegourd",
        "ring",
        "scale",
        "sculpture",
        "sheep",
        "shirts",
        "shoestand",
        "shoppingtrolley",
        "signboard",
        "sink",
        "sketchpen",
        "slippers",
        "soap",
        "soapdispenser",
        "socks",
        "sofa",
        "sofachair",
        "spectacle",
        "spoon",
        "sportsshoes",
        "stairs",
        "stamppad",
        "stapler",
        "staplerpins",
        "steelbowl",
        "steelplates",
        "stickynotes",
        "stool",
        "switchboard",
        "tablefan",
        "tankerlorry",
        "tap",
        "tape",
        "television",
        "tenrupeecoin",
        "thermoflask",
        "tomato",
        "toothbrush",
        "toothpaste",
        "tractor",
        "trashcan",
        "tray",
        "tree",
        "trolleybag",
        "tshirts",
        "tubelight",
        "tvcabinet",
        "tworupeecoin",
        "umbrellas",
        "wallbulb",
        "wallclock",
        "wallets",
        "washbasin",
        "watches",
        "waterbottles",
        "watermelon",
        "westerncommode",
        "window",
        "wineglass",
      ]
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # input details
    print(input_details)
    # output details
    print(output_details)
    img = np.asarray(image_path)
    new_img1 = cv2.resize(img, (224, 224))
    # images.append(new_img)
    new_img2 = new_img1/255
    new_img = np.float32(new_img2)
    # resize the input tensor
    input_tensor = np.array(np.expand_dims(new_img, 0))
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    predict_label = np.argsort(output_data[0])[:-10:-1]
    print(predict_label)
    print("###############", output_data[0][predict_label])
    result = []
    for i in range(len(predict_label)):
        result.append(classes[predict_label[i]])
    return result


# if __name__ == "__main__":
#     multilabel_main()
