from inspect import Parameter
from pandas import array
import pyocr
from unittest import case
from cv2 import BORDER_ISOLATED, BORDER_TRANSPARENT
import pytesseract
from PIL import ImageEnhance, ImageStat, Image as Img
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
import time
from skimage.restoration import estimate_sigma

# import skimage
# from skimage import filters
# from skimage.filters import threshold_local

class Tunable_Filter():
    def __init__(self, filter_image, operations, show = False):
        self._filter_image = filter_image
        self.operations = operations
        self.current_operation = 0
        self.show = show
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
        
    @property
    def filter_image(self):
        return self._filter_image
    
    @filter_image.setter
    def filter_image(self, value, *args, **kwargs):
        self._filter_image = value
        self._update_observers()
        
    def _update_observers(self):
        for observer in self.observers:
            observer()     
    
    def getangle(self, im):
        ''' looking for contours, calculates its angle to horizont
        and retuns result'''
        
        # return -0.99
        img = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_REFLECT)
        # im_show('border_img', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        erode_img = cv2.erode(blur_gray, kernel, iterations=8)
        # im_show('erode_img', erode_img)
        contours, hierarchy = cv2.findContours(erode_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i, angle = 0, 0
        for c in contours:
            if hierarchy[0][i][3] == 0:
                xy, wh, angle = cv2.minAreaRect(c)
                break
            i += 1

        if angle > 0:
            return angle -90 if angle > 60 else angle  
        else:
            return angle + 90 if angle < -60 else angle
                        
    def histogram(self):
        im_file = self.filter_image
        if isinstance(im_file, np.ndarray):
            im_file = Img.fromarray(im_file)
    
        return im_file.histogram()    
     
    def next_op(self):
        self.current_operation += 1

    def get_op(self):
        self.next_op()
        return self.operations[self.current_operation - 1] 

    def do_operation(self, digits):
        
        op = self.get_op()
        if op == 'enhance':
            self.filter_image = Img.fromarray(self.filter_image) if isinstance(self.filter_image, np.ndarray) else self.filter_image
            enhancer = ImageEnhance.Contrast(self.filter_image)
            self.filter_image = enhancer.enhance(2)
        elif op == 'contr_brigh':
            # self.pillow_adj(0.321, 0.8296659)
            self.filter_image = pillow_adj(self.filter_image, 0.421, 0.9)
        elif op == 'denoise':
            img = self.filter_image
            # self.filter_image = cv2.fastNlMeansDenoisingColored(img, None, 40 if digits else 80 , 10, 7, 21) #Андрианов
            self.filter_image = cv2.fastNlMeansDenoisingColored(img, None, 10 if digits else 10 , 10, 7, 21) # Ignatov
        elif op == 'negative':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)     
            self.filter_image = cv2.bitwise_not(self.filter_image)
        elif op == 'rotate':
            filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            angle = self.getangle(filter_image)
            if not (angle == 0):                
                filter_image = Img.fromarray(filter_image)
                self.filter_image = filter_image.rotate(angle)
                # im_show('rotated', self.filter_image)
        elif op == 'pixel':
            self.filter_image = Img.fromarray(self.filter_image) if isinstance(self.filter_image, np.ndarray) else self.filter_image
            width, height = self.filter_image.size
            for x in range(width):
                for y in range(height):
                    color = self.filter_image.getpixel((x,y))
                    new_color = color if color[0] + color[1] + color[2] < 171 else (255, 255, 255)
                    self.filter_image.putpixel((x, y), new_color)
        elif op == 'black_white':
            self.filter_image = Img.fromarray(self.filter_image) if isinstance(self.filter_image, np.ndarray) else self.filter_image
            thresh =  117
            fn = lambda x : 255 if x > thresh else 0
            self.filter_image = self.filter_image.convert('L').point(fn, mode='1')
        elif op == 'gray':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            self.filter_image = cv2.cvtColor(self.filter_image, cv2.COLOR_BGR2GRAY)
        elif op == 'medianBlur':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            self.filter_image = cv2.medianBlur(self.filter_image, 3)
        elif op == 'adaptiveThreshold':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            self.filter_image = cv2.adaptiveThreshold(self.filter_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,25)
        elif op == 'dilate':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
            self.filter_image = cv2.dilate(self.filter_image, kernel, iterations=4)
        elif op == 'erode':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            self.filter_image = cv2.erode(self.filter_image, kernel, iterations=1)            
        elif op == 'bilateral':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)
            self.filter_image = cv2.bilateralFilter(self.filter_image, 9, 20, 20, BORDER_TRANSPARENT)
        elif op == 'morphologyEx':
            self.filter_image = self.filter_image if isinstance(self.filter_image, np.ndarray) else np.array(self.filter_image)           
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            self.filter_image = cv2.morphologyEx(self.filter_image, cv2.MORPH_OPEN, kernel, iterations=5)
        elif op == 'apply_threshold':
            num = 7
            self.filter_image = apply_threshold(self.filter_image, num) # argment[1:7]
            op += '_' + str(num)
        elif op == 'resize':
            scale_percent = 240
            image = self.filter_image

            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            self.filter_image = cv2.resize(self.filter_image, dim, interpolation = cv2.INTER_AREA)
        elif op == 'background':
            backSub = cv2.createBackgroundSubtractorMOG2()
            fgMask = backSub.apply(np.float32(self.filter_image))
            # self.filter_image = fgMask #Поплавский без обработки фона
        if self.show:
            im_show(op, self.filter_image)
            if op == 'erode':
                img = Img.fromarray(self.filter_image) if isinstance(self.filter_image, np.ndarray) else self.filter_image
                img.save(op + '.png')
        
class Console_observer:
    def __init__(self, tf):
        self.tunable_filter = tf
        
    def __call__(self):
        if isinstance(self.tunable_filter._filter_image, np.ndarray):
            print(self.tunable_filter.current_operation, self.tunable_filter._filter_image.shape)

        
def img_filter(height, width, image):

    enhancer = ImageEnhance.Contrast(Img.fromarray(image))
    img2 = enhancer.enhance(4)
    img_neg = cv2.bitwise_not(np.array(img2))

    thresh = 100
    fn = lambda x : 255 if x > thresh else 0
    res = Img.fromarray(img_neg).convert('L').point(fn, mode='1')

    return res

def brightness_calc(im_file, num = 3):
    if isinstance(im_file, np.ndarray):
        im_file = Img.fromarray(im_file)

    stat = ImageStat.Stat(im_file)
    return round(stat.rms[0]/255, num)
    
def contrast_calc(im_file, num = 3):
    if isinstance(im_file, np.ndarray):
        im_file = Img.fromarray(im_file) 
        
    stat = ImageStat.Stat(im_file)         
    return round(stat.stddev[0]/255, num)

def enhance(image):
    # im_show('original', image) 
    image = Img.fromarray(image) if isinstance(image, np.ndarray) else image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # im_show('enhance', image)
    return np.array(image)
    
def pillow_adj(im, contrast, brightness):
    if isinstance(im, np.ndarray):
        im = Img.fromarray(im) 
    # im_show('original', im)       
    im = ImageEnhance.Brightness(im).enhance(brightness / brightness_calc(im))
    # im_show('brightness', im)  
    im = np.array(ImageEnhance.Contrast(im).enhance(contrast / contrast_calc(im)))
    # im_show('contrast', im) 
    return im
            
def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (1, 1), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 1), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 1),
    }
    return switcher.get(argument, "Invalid method")

def estimate_noise(img, multichannel=True, average_sigmas=True):
    img = img if isinstance(img, np.ndarray) else np.array(img)
    noise = estimate_sigma(img, multichannel=multichannel, average_sigmas=average_sigmas)
    if noise == None:
        return 0
    return round(noise, 3)

def get_text_json():
    pass

def tune_get_txt(img, lang = 'eng', show = False, digits = False):
    if show:
        Img.fromarray(img).save('inputImg.png')
    # im_show('before', img)
    # filter_list = ['bilateral', 'enhance', 'dilate', 'erode', 'gray', 'adaptiveThreshold', 'negative', 'black_white']
    # filter_list = ['bilateral', 'enhance', 'dilate', 'erode', 'morphologyEx', 'negative', 'black_white']
    # filter_list = ['bilateral', 'enhance', 'dilate', 'erode', 'gray', 'apply_threshold', 'negative', 'black_white']
    # filter_list = ['bilateral', 'enhance', 'dilate', 'erode', 'negative', 'black_white']
    # filter_list = ['resize', 'bilateral', 'enhance', 'dilate', 'erode', 'negative', 'black_white']
    sigma = estimate_noise(img)
    if sigma < 2:
        # Поплавский, Уфимцева, Игнатов - denoise added
        # filter_list = ['contr_brigh', 'denoise', 'resize', 'bilateral', 'enhance', 'dilate', 'erode', 'rotate', 'negative', 'black_white']
        # filter_list = ['contr_brigh', 'denoise', 'resize', 'bilateral', 'enhance',  'erode', 'rotate', 'negative', 'black_white']
        # filter_list = ['contr_brigh', 'denoise', 'bilateral', 'enhance',  'erode', 'rotate', 'negative', 'black_white']
        filter_list = ['contr_brigh', 'resize', 'denoise', 'enhance', 'rotate', 'negative', 'black_white']
    else:
        filter_list = ['denoise', 'contr_brigh', 'resize', 'rotate', 'background', 'negative', 'black_white']
    tunable = Tunable_Filter(img, filter_list, show=show)
    cons = Console_observer(tunable)
    # tunable.attach(cons)
    
    for op in filter_list:
        if show:
            print(op)
        tunable.do_operation(digits)
    tunable.filter_image = Img.fromarray(tunable.filter_image) if isinstance(tunable.filter_image, np.ndarray) else tunable.filter_image
    
    # plt.plot(tunable.histogram())
    # plt.show()
        
    if show:
        im_show('date' if digits else 'string', tunable.filter_image)
    if not digits:
        #pytesseract
        config = '--psm 8' if digits else '--psm 6' + ' --oem 1'
        txt = pytesseract.image_to_string(tunable.filter_image, lang=lang, config=config)
    else:
        # pyocr
        builder = pyocr.builders.DigitBuilder() if digits else pyocr.builders.TextBuilder()
        tools = pyocr.get_available_tools()[0]                               
        txt = tools.image_to_string(tunable.filter_image, lang=lang, builder=builder)
     
    return txt_format(txt, digits=digits)

def txt_format(txt, digits):
    if digits:
        try:
          obj = time.strptime(txt[:10], "%d.%m.%Y")
          return time.strftime('%Y%m%d', obj)
        except:
          return ''
    else:
        reg = re.compile('[^а-яА-Я]')
        txt = reg.sub('', txt)
        txt = txt.capitalize()        
    return txt

def get_txt(img, lang = 'eng'):
    # im_show('before', img)
    img_neg = img_filter(3, 3, img)
    # im_show('after', img_neg)

    return pytesseract.image_to_string(img_neg, lang=lang, config='--psm 6')

def im_show(what, img):
    if isinstance(img, np.ndarray):
        img = Img.fromarray(img)

    plt.title(what + ': brightness = {}, contrast = {}, noise = {}'.format(brightness_calc(img), contrast_calc(img),
              estimate_noise(img)))
    plt.imshow(img, interpolation='nearest')
    plt.show()          

