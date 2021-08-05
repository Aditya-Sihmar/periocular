import os
import cv2
import numpy as np
import pandas as pd
import skimage
import scipy
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS


class OCLBCP:
    """ This class uses the path of the image and calculates the 
      Local Binary Pattern of an image 
    
      It will take the homomorphic filterd image as the input and 
      calculates the LBP of the image and return it.
    
    """


    #initializing the variables
    def __init__(self, img,thres = 5):
        self.img = img
        self.thres = thres

    def ortho_combi(self):
        def get_pixel(img, center, x, y):
            
            lbp = 0
            ltp = 0            
            try:
                # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
                if img[x][y] >= center :
                    lbp = 1
                if img[x][y] >= center + self.thres:
                    ltp = 1
                elif img[x][y] <= center - self.thres:
                    ltp = -1
                else :
                    pass
                    
            except:
                # Exception is required when neighbourhood value of a center pixel value is null i.e. values present at boundaries.
                pass
            
            return lbp , ltp

        # Function for calculating LBP
        def ltp_calculated_pixel(img, x, y):

            center = img[x][y]

            lbp_arr = []
            pos_ltp = []
            neg_ltp = []
            
            # top_left
            lbp , ltp = get_pixel(img, center, x-1, y-1)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)
            
            # top
            lbp , ltp = get_pixel(img, center, x-1, y)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)
            
            # top_right
            
            lbp , ltp = get_pixel(img, center, x-1, y + 1)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)
            
            # right
            lbp , ltp = get_pixel(img, center, x, y + 1)
            lbp_arr.append(lbp < 0)
            if (ltp):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)
            
            # bottom_right
            lbp , ltp = get_pixel(img, center, x + 1, y + 1)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)

            # bottom
            lbp , ltp = get_pixel(img, center, x + 1, y)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)

            # bottom_left
            lbp , ltp = get_pixel(img, center,  x + 1, y-1)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)

            # left
            lbp , ltp = get_pixel(img, center, x, y-1)
            lbp_arr.append(lbp)
            if (ltp < 0):
                pos_ltp.append(0)
                neg_ltp.append(abs(ltp))
            else: 
                pos_ltp.append(ltp)
                neg_ltp.append(0)

            # print(f'Negative LTP {neg_ltp}')
            # print(f'positive LTR {pos_ltp}')

            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]
            a1 = [lbp_arr[0], pos_ltp[1], lbp_arr[2], pos_ltp[3], lbp_arr[4], pos_ltp[5],lbp_arr[6], pos_ltp[7]]
            a2 = [pos_ltp[0], lbp_arr[1], pos_ltp[2], lbp_arr[3], pos_ltp[4],lbp_arr[5], pos_ltp[6], lbp_arr[7]]
            a3 = [lbp_arr[0], neg_ltp[1], lbp_arr[2], neg_ltp[3], lbp_arr[4], neg_ltp[5],lbp_arr[6], neg_ltp[7]]
            a4 = [neg_ltp[0], lbp_arr[1], neg_ltp[2], lbp_arr[3], neg_ltp[4],lbp_arr[5], neg_ltp[6], lbp_arr[7]]

            orth = [ max(a1[i],a2[i],a3[i],a4[i]) for i in range(len(a1))]

            orth_val = 0
            
            for i in range(len(orth)):
                orth_val += orth[i] * power_val[i]
                
            return orth_val

        # img_bgr = cv2.imread(self.path, 1)

        height, width = img.shape

        # # We need to convert RGB image into gray one because gray image has one channel only.
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create a numpy array as the same height and width of RGB image
        img_orth = np.zeros((height, width),np.uint8)
    

        for i in range(0, height):
            for j in range(0, width):
                # img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                img_orth[i, j] = ltp_calculated_pixel(img, i, j)

        # plt.imshow(img)
        # plt.show()

        # plt.imshow(img_lbp, cmap ="gray")
        # plt.show()
        return img_orth

        print("LBP Program is finished")

class LocalTerPattern:
    """ This class uses the path of the image and calculates the 
      Local Binary Pattern of an image 
    
      It will take the homomorphic filterd image as the input and 
      calculates the LBP of the image and return it.
    
    """


    #initializing the variables
    def __init__(self, img,thres = 5):
        self.img = img
        self.thres = thres

    def ltp(self):
        def get_pixel(img, center, x, y):
            
            ltp_value = 0
            
            try:
                # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
                if img[x][y] >= center + self.thres:
                    ltp_value = 1
                if img[x][y] <= center - self.thres:
                    ltp_value = -1
            

                    
            except:
                # Exception is required when neighbourhood value of a center pixel value is null i.e. values present at boundaries.
                pass
            
            return ltp_value

        # Function for calculating LBP
        def ltp_calculated_pixel(img, x, y):

            center = img[x][y]

            val_ar = []
            pos_ltp = []
            neg_ltp = []
            
            # top_left
            val_ar.append(get_pixel(img, center, x-1, y-1))
            #print(f'center value {center} pixel val {get_pixel(img, center, x-1, y-1)}')
            if (get_pixel(img, center, x-1, y-1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x-1, y-1))
                #print('appending neg')
            else: 
                pos_ltp.append(get_pixel(img, center, x-1, y-1) )
                neg_ltp.append(0)
                #print('appending pos')
                #print(pos_ltp)
            
            # top
            val_ar.append(get_pixel(img, center, x-1, y))
            if (get_pixel(img, center, x-1, y) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x-1, y))
            else :
                pos_ltp.append(get_pixel(img, center, x-1, y) )
                neg_ltp.append(0)
            
            # top_right
            val_ar.append(get_pixel(img, center, x-1, y + 1))
            if (get_pixel(img, center, x-1, y + 1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x-1, y + 1))
            else :
                pos_ltp.append(get_pixel(img, center, x-1, y + 1) )
                neg_ltp.append(0)
            
            # right
            val_ar.append(get_pixel(img, center, x, y + 1))
            if (get_pixel(img, center, x, y + 1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x, y + 1))
            else :
                pos_ltp.append(get_pixel(img, center, x, y + 1) )
                neg_ltp.append(0)
            
            # bottom_right
            val_ar.append(get_pixel(img, center, x + 1, y + 1))
            if (get_pixel(img, center, x + 1, y + 1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x + 1, y + 1))
            else :
                pos_ltp.append(get_pixel(img, center,x + 1, y + 1) )
                neg_ltp.append(0)
            
            # bottom
            val_ar.append(get_pixel(img, center, x + 1, y))
            if (get_pixel(img, center,  x + 1, y) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center,  x + 1, y))
            else :
                pos_ltp.append(get_pixel(img, center,  x + 1, y) )
                neg_ltp.append(0)
            
            # bottom_left
            val_ar.append(get_pixel(img, center, x + 1, y-1))
            if (get_pixel(img, center, x + 1, y-1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center,  x + 1, y-1))
            else :
                pos_ltp.append(get_pixel(img, center,  x + 1, y-1) )
                neg_ltp.append(0)
            
            # left
            val_ar.append(get_pixel(img, center, x, y-1))
            if (get_pixel(img, center, x, y-1) < 0):
                pos_ltp.append(0)
                neg_ltp.append(get_pixel(img, center, x, y-1))
            else :
                pos_ltp.append(get_pixel(img, center, x, y-1) )
                neg_ltp.append(0)
            
            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]

            val = 0
            pos_val = 0
            neg_val = 0
            #print(neg_ltp)
            
            for i in range(8):
                val += val_ar[i] * power_val[i]
                pos_val += pos_ltp[i]*power_val[i]
                neg_val += neg_ltp[i]*power_val[i]
                
            return val, pos_val, neg_val

        # img_bgr = cv2.imread(self.path, 1)

        height, width = img.shape

        # # We need to convert RGB image into gray one because gray image has one channel only.
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create a numpy array as the same height and width of RGB image
        img_ltp = np.zeros((height, width),np.uint8)
        img_pos_ltp = np.zeros((height, width),np.uint8)
        img_neg_ltp = np.zeros((height, width),np.uint8)


        for i in range(0, height):
            for j in range(0, width):
                # img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                img_ltp[i, j],img_pos_ltp[i, j],img_neg_ltp[i, j] = ltp_calculated_pixel(img, i, j)

        # plt.imshow(img)
        # plt.show()

        # plt.imshow(img_lbp, cmap ="gray")
        # plt.show()
        return img_ltp,img_pos_ltp,img_neg_ltp

        print("LBP Program is finished")

class LocalBinPattern:
    """ This class uses the path of the image and calculates the 
      Local Binary Pattern of an image 
    
      It will take the homomorphic filterd image as the input and 
      calculates the LBP of the image and return it.
    
    """


    #initializing the variables
    def __init__(self, img):
        self.img = img
    def lbp(self):
        def get_pixel(img, center, x, y):
            
            new_value = 0
            
            try:
                # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
                if img[x][y] >= center:
                    new_value = 1
                    
            except:
                # Exception is required when neighbourhood value of a center pixel value is null i.e. values present at boundaries.
                pass
            
            return new_value

        # Function for calculating LBP
        def lbp_calculated_pixel(img, x, y):

            center = img[x][y]

            val_ar = []
            
            # top_left
            val_ar.append(get_pixel(img, center, x-1, y-1))
            
            # top
            val_ar.append(get_pixel(img, center, x-1, y))
            
            # top_right
            val_ar.append(get_pixel(img, center, x-1, y + 1))
            
            # right
            val_ar.append(get_pixel(img, center, x, y + 1))
            
            # bottom_right
            val_ar.append(get_pixel(img, center, x + 1, y + 1))
            
            # bottom
            val_ar.append(get_pixel(img, center, x + 1, y))
            
            # bottom_left
            val_ar.append(get_pixel(img, center, x + 1, y-1))
            
            # left
            val_ar.append(get_pixel(img, center, x, y-1))
            
            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]

            val = 0
            
            for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]
                
            return val

        # img_bgr = cv2.imread(self.path, 1)

        height, width = img.shape

        # # We need to convert RGB image into gray one because gray image has one channel only.
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create a numpy array as the same height and width of RGB image
        img_lbp = np.zeros((height, width),np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                # img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                img_lbp[i, j] = lbp_calculated_pixel(img, i, j)

        # plt.imshow(img)
        # plt.show()

        # plt.imshow(img_lbp, cmap ="gray")
        # plt.show()
        return img_lbp

        print("LBP Program is finished")