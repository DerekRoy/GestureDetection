import numpy as np
import imutils
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
from keras.models import load_model


#Set up the models to find hands and predict the gesture
model = load_model("./models/raw28k(pyimage)5hrs.h5py")
detection_graph, sess = detector_utils.load_inference_graph()

#Get the front facing camera and height andf width
camera = cv2.VideoCapture(0)
im_width, im_height = (camera.get(3), camera.get(4))

#Label handling and color assignments
GESTURES = ["forks", "left", "stop", "peace", "right"]
colors = {"forks":(82,240,247),"left":(90,153,219),"stop":(86,127,113),"peace":(7,24,90),"right":(93,90,90)}

#Detected number of hands
num_hands_detect = 1

#Skin tone
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

#Change to True to save hand images
createData = False

#Collecting Data
i = 5000
train_size = 5741
gesture = "stop"
path = "./Data/{}/{}(".format(gesture,gesture)


while True:
    #Get the frame from the camera, flip, and convert to RGB
    (grabbed,frame) = camera.read()
    frame = cv2.flip(frame,1)
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Create canvases of gesture detection
    canvas = np.full((300, 400, 3), 255, dtype="uint8")

    #Draw Title text for two canvases
    cv2.putText(canvas, "Gestures Detected", (69, 28),cv2.FONT_HERSHEY_COMPLEX, 0.80,(0, 0, 0), 2)
    cv2.putText(canvas, "Gesture", (5, 68),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(5, 5, 5), 2)
    cv2.putText(canvas, "LIKELINESS", (266, 68),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(5, 5, 5), 2)

    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

    #Draw bounding boxes on frame
    x = detector_utils.draw_box_on_image(num_hands_detect, .2, scores, boxes, im_width, im_height,image_np)

    if len(x) > 0:
        x = x[0]
        #Get int index values for the image
        l,r,t,b = int(x[0]), int(x[1]), int(x[2]), int(x[3])

        #Get hand from image and process to put into the model
        im = frame[t:b,l:r]

        #Skin Color based method of tracking hands
        #Convert to HSV and then apply the average skin color to make mask
        # blur the mask to help remove noise
        # converted = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # skinMask = cv2.inRange(converted, lower, upper)
        # skinMask = cv2.GaussianBlur(skinMask, (7, 7), 0)
        skinMask = im

        #This chunk of code was used to get training data for the model
        #Just rename the path and then start the code and make the sign you want to get data on
        if createData:
            i +=1
            # skinMask = cv2.resize(skinMask, (50, 50))
            skinMask = cv2.resize(im, (50,50))
            print("Data Image: {}({})".format(gesture, i))
            cv2.imshow("mask", skinMask)
            # ycrcb = process(im)
            # cv2.imshow("ycrbc", ycrcb*255)
            cv2.imwrite(path+"{}).jpg".format(i),skinMask)
            if i >train_size:
                break;

        else:
            #Prepare to predict
            sm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            sm = cv2.resize(sm, (50, 50))
            sm = np.asarray(sm)/255
            sm = np.expand_dims(sm,axis=2)
            # ycrcb = process(im)
            preds = model.predict(np.expand_dims(sm, axis = 0))[0]

            #Get label of most likely value
            most_likely = GESTURES[np.argmax(preds)]

            #Draw label for the box
            cv2.putText(image_np, most_likely, (l, t - 10),cv2.FONT_HERSHEY_SIMPLEX, 3, (77, 255, 9), 2)

            # loop over the labels + probabilities and draw them
            for (j, (gesture, prob)) in enumerate(zip(GESTURES, preds)):
                # construct the label text
                text = gesture
                text2 = "{:.0f}%".format(prob * 100)

                #Draw the label + probability bar on the canvas
                w = int(prob * 220)
                cv2.rectangle(canvas, (75, (j * 35) + 80),(75+w, (j * 35) + 110), (colors[gesture]), -1)
                cv2.putText(canvas, text, (5, (j * 35) + 98),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(5, 5, 5), 2)
                cv2.putText(canvas, text2, (300, (j * 35) + 98),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(5, 5, 5), 2)


    #Below will have to be rewritten
    image_np = imutils.resize(image_np, height=300)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imshow('Single-Threaded Detection', image_np)
    #Create the graph of detection
    if not createData:
        cv2.imshow('Scores', canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
