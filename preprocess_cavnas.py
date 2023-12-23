import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x0 = [-1, 1, 0, 0]
y0 = [0, 0, -1, 1]

class ImageProcessor():

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
        #Dfs to find every digits on canvas
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
                    
                    self.dfs(img, m, n, i, j, self.num_label)
                    self.contours.append((self.min_x, self.min_y, self.max_x, self.max_y))
                    
        #Adding bounding boxes of each digit to contour
        
        data, img_data = self.digit_segmentation(img, self.num_label, self.contours)
        self.processed_img, self.unflattened_img = data, img_data
        
    def digit_segmentation(self, a, num_label, contours):
        
        '''
        res: array of (784, 1) size vectors
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
            
            #Reshape the processed image then append it
            res2.append(processed_img.reshape(28, 28, 1))
            image_np = processed_img.reshape(1, 784).squeeze()
            res.append(image_np)

        return np.array(res), np.array(res2)
    
    def display(self):
        fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (12, 12))
        cur_label = 0
        for i in range(3):
            for j in range(3):
                if cur_label == self.num_label:
                    continue
                ax[i, j].imshow(self.unflattened_img[cur_label], cmap = 'gray')
                ax[i, j].axis("off")
                
                cur_label += 1
        plt.show()

def resize_my_version(img):
    height, width = img.shape

    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((x,y), np.uint8)
    square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
    image_resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_LINEAR)
    padding_horizontal = np.zeros((image_resized.shape[0], 4)).astype(np.uint8)
    padding_vertical = np.zeros((4, 28)).astype(np.uint8)
    image_resized = np.concatenate((padding_horizontal, image_resized, padding_horizontal), axis = 1)
    image_resized = np.concatenate((padding_vertical, image_resized, padding_vertical), axis = 0)
    return image_resized

