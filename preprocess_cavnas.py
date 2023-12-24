import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x0 = [-1, 1, 0, 0]
y0 = [0, 0, -1, 1]

class CanvasProcessor():

    def __init__(self):
        '''
        variables:
            vst[i][j] = 1 if pixel located at (i, j) is visited
            num_label: Number of separated regions on the canvas
            min_x, min_y, max_x, max_y: upper left corner and lower right corner of 
                                        each digit's bounding box, for temporary use only
            contours: Save the left corner and lower right corner of each digits
            unflattened_img: return (28, 28) image instead of (1, 784) vector, for CNN only
            processed_img: return an array of (1, 784) vectors
        '''
        self.processed_img = None
        self.vst = None
        self.num_label = 0
        self.min_x, self.min_y = float('inf'), float('inf')
        self.max_x, self.max_y = 0, 0
        
        self.contours = None
        self.unflattened_img = None
        
    def dfs(self, a, m, n, x, y, label):
        '''
        METHOD DESCRIPTION:
        DFS on a digit to locate upper left corner and lower right corner of that digit
        Through (self.min_x, self.min_y) and (self.max_x, self.max_y) To find the bounding 
        box
        
        Variables:
            a: canvas pixel matrix
            m, n: Size of canvas
            x, y: starting location for DFS
        '''
        global x0
        global y0
        #Initialize a stack
        st = []
        st.append((x, y))
        
        while(len(st) > 0):
            curr = st[len(st) - 1]
            st.remove(st[len(st) - 1])

            row, col = curr[0], curr[1]
            
            #Finding left upper corner and right lower corner
            self.min_x = min(self.min_x, row)
            self.max_x = max(self.max_x, row)

            self.min_y = min(self.min_y, col)
            self.max_y = max(self.max_y, col)

            #Looking for possible neighbors
            for i in range(4):
                x_ = row + x0[i]
                y_ = col + y0[i]

                if x_ < 0 or x_ >= m or y_ < 0 or y_ >= n or a[x_, y_] == 0:
                    continue
                if self.vst[x_, y_] != 0:
                    continue
                self.vst[x_, y_] = label
                st.append((x_, y_))
    
    #Fit the image from canvas to process, separating into different 28x28 pixel images
    def fit(self, img):
        '''
        METHOD DESCRIPTION:
        Perform processing step on the image retrieved from canvas.
        Save those data in:
            self.processed_img: array of (, 784) vectors
            self.unflattened_img: array of (28, 28, 1) size image
    
        Variables: 
            img: Image retrieved from canvas
        '''
        
        #Initialize variables before performing DFS
        m, n = img.shape
        self.vst = np.zeros(img.shape)
        self.num_label = 0
        self.contours = []
        
        #Perform DFS search on canvas
        for i in range(m):
            for j in range(n):
                if img[i, j] != 0 and self.vst[i, j] == 0:
                    self.min_x, self.min_y = float('inf'), float('inf')
                    self.max_x, self.max_y = 0, 0
                    self.num_label += 1
                    
                    #Find bounding box via DFS
                    self.dfs(img, m, n, i, j, self.num_label)
                    self.contours.append((self.min_x, self.min_y, self.max_x, self.max_y))
                    
        #Perform digit segmentation to take out array of (, 784) vectors and array of (28, 28, 1) images
        data, img_data = self.digit_segmentation(img, self.num_label, self.contours)
        self.processed_img, self.unflattened_img = data, img_data
        
    def digit_segmentation(self, a, num_label, contours):
        '''
        METHOD DESCRIPTION:
            Performing preprocess each digit to fit MNIST format.
        Variables:
            a: pixel matrix retrieved from canvas
            num_label: Number of digits
            contours: Array of bounding box coordinates

            res: array of (, 784) size vectors
            res2: array of (28, 28, 1) size images
        Return:
            res: array of (, 784) size vectors
            res2: array of (28, 28, 1) size images
        '''

        res = []
        res2 = []

        for i in range(num_label):
            #Geting bounding box of each digit
            x0 = contours[i][0]
            y0 = contours[i][1]
            x1 = contours[i][2]
            y1 = contours[i][3]

            #Get the image out of it
            img = a[x0:x1, y0:y1].astype(np.uint8)

            #Put it into a square image
            processed_img = resize_my_version(img)
            
            #Reshape the processed image then append it to return arrays
            res2.append(processed_img.reshape(28, 28, 1))
            image_np = processed_img.reshape(1, 784).squeeze()
            res.append(image_np)

        return np.array(res), np.array(res2)

def resize_my_version(img):
    '''
    METHOD DESCRIPTION:
    Resizing img to (28, 28) shape without distortion
    
    Variables:
        img: Image of the digit
    Return:
        image_resized: Resized image of that digit
    '''
    height, width = img.shape
    
    #Get square of zeros (black color)
    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((x,y), np.uint8)
    
    #Add the given image of digit to the center square
    square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
    
    #Resize image to (20, 20)
    image_resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_LINEAR)
    
    #Add padding of 4 to image so that it has size (28, 28)
    padding_horizontal = np.zeros((image_resized.shape[0], 4)).astype(np.uint8)
    padding_vertical = np.zeros((4, 28)).astype(np.uint8)
    image_resized = np.concatenate((padding_horizontal, image_resized, padding_horizontal), axis = 1)
    image_resized = np.concatenate((padding_vertical, image_resized, padding_vertical), axis = 0)
    return image_resized

