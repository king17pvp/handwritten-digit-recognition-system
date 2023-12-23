from tkinter import *
import customtkinter as ct
import cv2
import numpy as np
import sklearn
from tensorflow.keras.models import load_model

from knn import *
from preprocess_cavnas import *
from feed_foward_model import *
from PIL import ImageGrab

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\', '/') + '/'

#Initializing models         
weight = np.load(dir_path + "weight/MNIST_ver3.npy", allow_pickle=True).tolist()
NN = Model()
NN.fit(weight, 4)
CNN = load_model(dir_path + "weight/final_model.h5")
KNN = KNNModel(7)
KNN.fit()

# Define colors
CANVAS_BG_COLOR = "black"
PEN_COLOR = "white"
ct.set_appearance_mode("black")


#Create a new window with fixed resolution
root = ct.CTk()
root.geometry("1280x720")
root.title("Recognizing Hand-written Digits")

#Create a canvas and a sidebar frame for drawing
canvas = ct.CTkCanvas(master=root, width=1000, height=720, bg=CANVAS_BG_COLOR)
canvas.pack(side="right", fill="both", expand=True)

sidebar_frame = ct.CTkFrame(master=root, width=280, corner_radius=10, fg_color=CANVAS_BG_COLOR)
sidebar_frame.pack(side="left", fill="y")



def clear_canvas():
    ''' 
    Clear the canvas
    '''
    canvas.delete("all")

# Bind mouse events to draw on canvas
last_x, last_y = 0, 0

def activate_event(event):
    global last_x, last_y
    canvas.bind("<B1-Motion>", draw_lines)
    last_x, last_y = event.x, event.y

def draw_lines(event):
    global last_x, last_y
    x, y = event.x, event.y
    canvas.create_line((last_x, last_y, x, y), width=10, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    last_x, last_y = x, y

def processCanvas(cv):
    '''
    METHOD DESCRIPTION:
    used to grab image from canvas, convert it into black and white images
    then feeding it into the ImageProcessor 
    
    return flattened images, original image, bounding box coordinates of 
    each digits in canvas and the processed unflattened images after seperating it from canvas
    '''
    widget = cv 
    
    #Getting the upper left corner and right lower corner of canvas
    x0 = widget.winfo_rootx() + 1.8
    y0 = widget.winfo_rooty() + 1.8
    x1 = x0 + widget.winfo_width() - 3.6
    y1 = y0 + widget.winfo_height() - 3.6
    
    #Grabbing the image out from the canvas
    img = np.array(ImageGrab.grab().crop((x0, y0, x1, y1)))
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(gray_img, kernel, iterations=4)    
    output_img = img.copy()
    
    #Feed it through the processor
    processor = ImageProcessor()
    processor.fit(img_dilation)
    
    processed_img = processor.processed_img
    contours = processor.contours
    unflattened_img = processor.unflattened_img
    
    return processed_img, contours, output_img, unflattened_img

def show_bounding_box():
    '''
    METHOD DESCRIPTION:
    Showing the bounding box of each digit in canvas after being processed
    '''
    processed_img, contours, output_img, unflattened_img = processCanvas(canvas)
    
    for i, rectangle in enumerate(contours):
        left_upper_point = rectangle[1], rectangle[0]
        right_lower_point = rectangle[3], rectangle[2]
        cv2.rectangle(output_img, left_upper_point, right_lower_point, (0, 255, 0), thickness=5)
        

    cv2.imshow('Bounding box', output_img)
    print("Bounding box displayed")
    
def recognize_digit_knn():
    '''
    METHOD DESCRIPTION:
    Predict the image using KNN
    '''
    processed_img, contours, output_img, unflattened_img = processCanvas(canvas)
    #pred = NN.predict(processed_img / 255)
    pred = KNN.predict(processed_img / 255)
    for i, rectangle in enumerate(contours):
        #print(unflattened_img[i].reshape(-1, 28, 28, 1).shape)
        final_pred = pred[i]
        
        left_upper_point = rectangle[1], rectangle[0]
        right_lower_point = rectangle[3], rectangle[2]
        output_str = str(final_pred)
        cv2.rectangle(output_img, left_upper_point, right_lower_point, (0, 255, 0), thickness=5)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(output_img, output_str, (rectangle[1], rectangle[0] - 5), font, fontScale, color, thickness)

    cv2.imshow('prediction by KNN', output_img)
    cv2.waitKey(0)
    print("Digit predicted using KNN")
    
def recognize_digit_cnn():
    '''
    METHOD DESCRIPTION:
    Predict the image using CNN
    '''
    processed_img, contours, output_img, unflattened_img = processCanvas(canvas)
    unflattened_img = unflattened_img.reshape(len(contours), 28, 28, 1)

    pred = CNN.predict(unflattened_img / 255)
    for i, rectangle in enumerate(contours):

        prob_distribution = pred[i]
        final_pred = np.argmax(prob_distribution)
        
        left_upper_point = rectangle[1], rectangle[0]
        right_lower_point = rectangle[3], rectangle[2]
        output_str = str(final_pred) + ' ' + str(int(max(prob_distribution) * 100)) + '%'
        cv2.rectangle(output_img, left_upper_point, right_lower_point, (0, 255, 0), thickness=5)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(output_img, output_str, (rectangle[1], rectangle[0] - 5), font, fontScale, color, thickness)

    cv2.imshow('prediction by CNN', output_img)
    cv2.waitKey(0)
    print("Digit predicted using CNN")
    
def recognize_digit_mlp():
    '''
    METHOD DESCRIPTION:
    Predict the image using MLP
    '''
    #Taking necessaries from processing canvas
    processed_img, contours, output_img, unflattened_img = processCanvas(canvas)
    pred = NN.predict(processed_img / 255)
    
    #Predict each number appeared in the canvas
    for i, rectangle in enumerate(contours):
        left_upper_point = rectangle[1], rectangle[0]
        right_lower_point = rectangle[3], rectangle[2]
        output_str = str(pred[0][i]) + " " + str(round(pred[1][i] * 100, 2)) + "%"
        cv2.rectangle(output_img, left_upper_point, right_lower_point, (0, 255, 0), thickness=5)
        
        #Draw the prediction and the probability
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(output_img, output_str, (rectangle[1], rectangle[0] - 5), font, fontScale, color, thickness)

    #Display the prediction
    cv2.imshow('prediction by MLP', output_img)
    cv2.waitKey(0)
    print("Digit predicted using MLP")

#First button for KNN prediction
button_1 = ct.CTkButton(master=sidebar_frame, text = "KNN Prediction", command=recognize_digit_knn, border_width=0, corner_radius=10)
button_1.pack(pady=30)

#Second button for MLP prediction
button_2 = ct.CTkButton(master=sidebar_frame, text = "MLP Prediction", command=recognize_digit_mlp, border_width=0, corner_radius=10)
button_2.pack(pady=30)

#Third button for CNN Prediction
button_3 = ct.CTkButton(master=sidebar_frame, text = "CNN Prediction", command=recognize_digit_cnn, border_width=0, corner_radius=10)
button_3.pack(pady=30)

#Fourth button for Clearing out canvas
button_4 = ct.CTkButton(master=sidebar_frame, text = "Clear canvas", command=clear_canvas, border_width=0, corner_radius=10)
button_4.pack(pady=30)

#Fifth button for showing bounding box
button_5 = ct.CTkButton(master=sidebar_frame, text = "Show box", command=show_bounding_box, border_width=0, corner_radius=10)
button_5.pack(pady=30)


canvas.bind("<Button-1>", activate_event)
root.mainloop()
