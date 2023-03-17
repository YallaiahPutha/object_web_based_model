

'''
Summary : For this file we called the Fruits detection model and got output as class_name and bounding_boxes
'''

import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

modelpath = "./tflite_models/fruit.tflite"

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Fruits_Detection')


def fruits(image_path, modelpath):
    """
    We are passing  image_path and model_path as input and got output as bounding_box coordinates and label index positions.
    Args:
        image_path (path): image file path
        modelpath (path): model file path

    Returns:
        Output (list of coordinates): got output bounding box coordinats and label index position
    """
    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    # Get model details
    input_details = interpreter.get_input_details()
    print("InputDetails:", input_details)
    output_details = interpreter.get_output_details()
    print("OUTputDetails:", output_details)
    # height = input_details[0]['shape'][1]
    # width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Load image and resize to expected shape [1xHxWx3]
    img = np.asarray(image_path)
    image_np = cv2.resize(img, (640, 640))
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image_rgb.shape
    input_data = np.expand_dims(image_rgb, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Retrieve detection resultsst.write(class_names[int(i-1)])
    # Bounding box coordinates of detected objects
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    # Class index of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    # Confidence of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    return boxes, scores, classes, imH, imW, image_rgb


def fruits_main():
    """
    This function we are calling the detection function and getting the user interface to upload the image bar available.
    """
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Detection")
    if file_uploaded is not None:
        image = np.array(Image.open(file_uploaded))
        orig_h, orig_w, _ = np.shape(image)
        print("In Main function", orig_h, orig_w)
        # test_img=load_img(image,target_size=(224,224))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        my_bar = st.progress(0)
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                boxes, scores, classes, imH, imW, image_rgb = fruits(image, modelpath)
                class_names = ["lemon", "cherry", "banana", "applered", "coconut", "guava", "pomegranate", "orange", "papaya", "kiwi", "mosambi", "grapewhite", "grapeblue", "watermelon", "avocado", "pineapple", "custardapple", "mango", "jamunfruit", "iceapple", "raspberry"]
                scores_indices = [idx for idx, score in enumerate(scores) if score > 0.50]
                classes = [int(classes[idx]) for idx in scores_indices]
                boxes = [boxes[idx] for idx in scores_indices]
                print(boxes)
                xmin, ymin, xmax, ymax, index = 0, 0, 0, 0, 0
                for index, box in enumerate(boxes):
                    ymin = int(box[0] * orig_h)
                    xmin = int(box[1] * orig_w)
                    ymax = int(box[2] * orig_h)
                    xmax = int(box[3] * orig_w)
                if index < len(boxes):
                    start_point = (xmin, ymin)
                    print(start_point)
                    end_point = (xmax, ymax)
                    print(end_point)
                    color = (0, 0, 255)
                    thickness = 5
                    cv2.rectangle(image, start_point, end_point, color, thickness)
                    cv2.putText(image, class_names[classes[index]], (xmin, ymin+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    st.write(class_names[classes[index]])
                    st.image(image, caption='Proccesed Image.')
                    cv2.waitKey(0)
                else:
                    st.write("No Predictions Found")
                cv2.destroyAllWindows()
                my_bar.progress(100)


# if __name__ == "__main__":
#     fruits_main()
