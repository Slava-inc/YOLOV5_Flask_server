from tkinter import Image
import cv2
import numpy as np
import fitz
from pathlib import Path
from PIL import Image as pil
import io
import img_to_txt as totext

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def convert_pdf(path):
    pages = fitz.open(path)
    ID3 = pages.getPageImageList(0)
    pixmap = fitz.Pixmap(pages, ID3[0][0])
    pixmap.gamma_with(1.1)
    pixmap.writeImage('tmp.png')
    with  pil.open('tmp.png') as img:
        img.save('tmp.jpg', dpi=(500, 500)) 
    return cv2.imread('tmp.jpg')

def binary_load(bin_file):
    pages = fitz.Document(stream=bin_file)
    ID3 = pages.getPageImageList(0)
    pixmap = fitz.Pixmap(pages, ID3[0][0])
    pixmap.gamma_with(1.1) 
    pixmap.writeImage('tmp.png')
    with  pil.open('tmp.png') as img:
        img.save('tmp.jpg', dpi=(500, 500)) 
    img = cv2.imread('tmp.jpg')
    
    cnts = looking_for_rectangles(img)
    img = getImage(cnts, img)
    
    # return img
    return img if img.shape[0] > img.shape[1] else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

def load_binary_image(bin_file):
    io_data = io.BytesIO(bin_file)
    img = pil.open(io_data)
    return np.array(img) 

# from ./cli Recognition
def looking_for_rectangles(image):
    
    height = 5
    width = 5

    # Источник: https://tonais.ru/library/izmenenie-razmera-izobrazheniya-opencv-cv2-v-python

    # Invert the image using cv2.bitwise_not
    img_neg = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # totext.im_show('gray', gray)
    blur = cv2.medianBlur(gray, 3)
    # totext.im_show('blur', blur)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,25)
    # totext.im_show('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (height,width))

    dilate = cv2.dilate(thresh, kernel, iterations=8)
    # totext.im_show('dilate', dilate)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

# looking for real image 
def getImage(rectangles, image):
        max_area = 0
        for c in rectangles:
            area = cv2.contourArea(c)
            if area > max_area: 
                max_area = area
                x,y,w,h = cv2.boundingRect(c)  

        return image[y:y+h, x:x+w] 
  