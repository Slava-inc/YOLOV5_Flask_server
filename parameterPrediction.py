import re
from string import digits
from sklearn import datasets
import tensorflow as tf
import pytesseract
import pyocr
# keras
from tensorflow import keras
import keras.layers as L
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_addons.layers import WeightNormalization
# from tensorflow.keras.activations import swish

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from skimage.restoration import estimate_sigma
import cv2
import numpy as np
from PIL import ImageEnhance, ImageStat, Image as Img
import pickle
from matplotlib import pyplot as plt
from cv2 import BORDER_ISOLATED, BORDER_TRANSPARENT
import pandas as pd
import img_to_txt as itt
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from plots import plot_one_box
from models.experimental import attempt_load
import torch
from dataloader import letterbox
import random
from io import StringIO


def round_to_odd(f):
    return int(np.round(f/2) * 2 + 1)


def get_image_array(imgs):
    # imgs - pillow images list
    images_train = []
    for image in imgs:
        assert (image is not None)
        image = np.array(image)
        image = cv2.resize(image, (255, 32))
        images_train.append(image)
    images_train = np.array(images_train)
    print('images shape', images_train.shape, 'dtype', images_train.dtype)
    return (images_train)


def tf_process_train_dataset_element(image, X, Y):
    return (image, X), Y


def tf_process_tabular_dataset_element(X, Y):
    return (X), Y


def plt_show(im, title='', num=1):
    plt.figure("birth date")
    plt.figure(num=num, figsize=(8, 5),)
    plt.title(title)
    plt.axis('off')  # Не отображать ось
    plt.imshow(im)


def brightness_calc(im_file, num=3):
    if isinstance(im_file, np.ndarray):
        im_file = Img.fromarray(im_file)

    stat = ImageStat.Stat(im_file)
    return round(stat.rms[0]/255, num)


def contrast_calc(im_file, num=3):
    if isinstance(im_file, np.ndarray):
        im_file = Img.fromarray(im_file)

    stat = ImageStat.Stat(im_file)
    return round(stat.stddev[0]/255, num)


def dictvalues_tolist(options):
    for item in options.items():
        options.update({item[0]: [item[1]]})
    return options


def estimate_noise(img, multichannel=True, average_sigmas=True):
    img = img if isinstance(img, np.ndarray) else np.array(img)
    noise = estimate_sigma(img, multichannel=multichannel,
                           average_sigmas=average_sigmas)
    if noise == None:
        return 0
    return round(noise, 3)


def bd_to_text(data):

    results = {}

    for item in data.items():
        name = re.sub(' ', '_', item[0])
        if name == 'birth_date':
            # Load model
            model = attempt_load('best.pt')
            digits = True
        else:
            # Load model
            model = attempt_load('best_cirillic.pt', map_location='cpu')
            digits = False
            # results[name] = ''
            # continue
        img0s = item[1]
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        stride = int(model.stride.max())  # model stride
        imgsz = 416
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # img boarding
        # h, w = img0s.shape[:2]
        # img0s = cv2.copyMakeBorder(img0s, int((imgsz - h)/2), int((imgsz - h)/2), int((imgsz - w)/2), int((imgsz - w)/2), cv2.BORDER_CONSTANT)

        # Padded resize
        img0s = letterbox(img0s, imgsz, stride=stride, scaleup=False)[0]

        # Convert
        img = img0s[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, multi_label=False)
        # Process detections
        str_data = "cls x y w h con\n"
        for i, det in enumerate(pred):  # detections per image
            # p = Path(p)
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            s = '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(img0s.shape)[[1, 0, 1, 0]]
            # print(' IN {0} write {1}'.format(txt_path, len(det)))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0s.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # if save_conf else (cls, *xywh)  # label format
                        line = (cls, *xywh, conf)
                        # print('{0} to file {1}'.format(line, txt_path + '.txt') )
                        # with open(txt_path + '.txt', 'a') as f:
                        str_data += ('%g ' * len(line)).rstrip() % line + '\n'
                        # label = f'{names[int(cls)]}'  # {conf:.2f}
                        # plot_one_box(xyxy, img0s, label=label,
                        #              color=colors[int(cls)], line_thickness=3)
                res = pd.read_csv(StringIO(str_data), sep=' ')
                t = torch.from_numpy(res.drop_duplicates().to_numpy())
                t = t[np.argsort(t[:, 1])]
                if digits:
                    converted = list(map(name_conv, t[:, 0].int()))
                else:
                    converted = list(map(cirillic_conv, t[:, 0].int()))
            results[name] = itt.txt_format(''.join(converted), digits)
        # else:
        #     results[name] = ''
    return results


def name_conv(a):
    names = ['0', '1', '.', '2', '3', '4', '5', '6', '7', '8', '9']
    return names[a]


def cirillic_conv(a):
    names = ['А', 'Б', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'В', 'У', 'Ф', 'Х',
             'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Ь', 'Г', 'Э', 'Ю', 'Я', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И']
    return names[a]


def images_to_text(data):

    images = data.values()

    result = {}
    img_df = pd.DataFrame(images, columns=['img'])

    X_columns = ['bright', 'contr', 'noise']

    img_df['bright'] = img_df['img'].apply(lambda x: brightness_calc(x))
    img_df['contr'] = img_df['img'].apply(lambda x: contrast_calc(x))
    img_df['noise'] = img_df['img'].apply(lambda x: estimate_noise(x))

    X_test = img_df[[x for x in X_columns]]
    # Scaling
    scaler = MinMaxScaler()
    for col in X_test.columns:
        X_test[col] = scaler.fit_transform(X_test[[col]])[:, 0]

    # images_test = get_image_array(list(img_df['img'].array))
    Y_test = np.zeros(len(X_test))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, Y_test)).map(tf_process_tabular_dataset_element)

    another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    with another_strategy.scope():
        model = tf.keras.models.load_model(
            'param_model_mse_b3_XY_no_noise.keras')
    test_predict_nn3 = model.predict(test_dataset.batch(30))
    with open('minmax.pickle', 'rb') as file:
        minmax = pickle.load(file)

    params = None
    rows, colums = test_predict_nn3.shape
    for i, ind in enumerate(img_df.index):
        # param = test_predict_nn3[i, :]
        for j in range(colums):
            mn = minmax[j, 1]
            mx = minmax[j, 0]
            # scaler.fit(np.array(img_df.iloc[[l for l in range(rows)], j+1]).reshape(-1, 1))
            # param = scaler.inverse_transform(test_predict_nn3)[:, i]
            param = test_predict_nn3[:, j].reshape(-1, 1) * (mx - mn) + mn
            if params is None:
                params = param.reshape(-1, 1)
            else:
                params = np.concatenate((params, param), axis=1)

        filters = [{'name': 'Adjust_brightness', 'param': {'brightness': np.round(params[i, 3], decimals=3)}},
                   {'name': 'Adjust_contrast', 'param': {
                       'contrast': np.round(params[i, 2], decimals=3)}},
                   # {'name': 'Resize', 'param': {'scale_percent': int(params[i, 1])}},
                   {'name': 'Resize', 'param': {}},
                   {'name': 'Enhance', 'param': {
                       'coef': np.round(params[i, 0], decimals=3)}},
                   {'name': 'Rotate', 'param': {}},
                   {'name': 'Negative', 'param': {}},
                   {'name': 'Black_white', 'param': {
                       'thresh': int(params[i, 4])}}
                   ]

        '''filters= [{'name': 'Adjust_brightness', 'param': {'brightness': 0.3}},
              {'name': 'Adjust_contrast', 'param': {'contrast': 2.}},
              # {'name': 'Resize', 'param': {'scale_percent': int(params[i, 1])}},
              {'name': 'Resize', 'param': {}},              
              {'name': 'Enhance', 'param': {'coef': 0.8}}, 
              {'name': 'Rotate', 'param': {}}, 
              {'name': 'Negative', 'param': {}},
              {'name': 'Black_white', 'param': {'thresh': 100}}
      ]'''

        nm = list(data.keys())[i]
        digits = (nm == 'birth date')
        tuner = Tuner_Instances()
        for filter in filters:
            tuner.construct(filter)

        image = img_df.loc[ind, 'img']
        img = tuner.filter_exec(image)

        if not digits:
            # pytesseract
            config = '--psm 8' if digits else '--psm 6' + ' --oem 1'
            txt = pytesseract.image_to_string(img, lang='rus', config=config)
        else:
            # pyocr
            builder = pyocr.builders.DigitBuilder() if digits else pyocr.builders.TextBuilder()
            tools = pyocr.get_available_tools()[0]
            txt = tools.image_to_string(Img.fromarray(img) if isinstance(
                img, np.ndarray) else img, lang='eng', builder=builder)
        result[nm] = itt.txt_format(txt, digits)
    return result


class Tuner:
    # ------------------------- filters ---------------------
    class Denoise:
        '''src	- Input 8-bit 3-channel image.
        dst	- Output image with the same size and type as src .
        templateWindowSize -	Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
        searchWindowSize	- Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. 
          Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
        h	- Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, 
          smaller h value preserves details but also preserves some noise
        hColor - The same as h but for color components. For most images value equals 10 will be enough to remove colored noise and do not distort 
          colors'''

        default_options = {'src': None, 'dst': None, 'templateWindowSize': 10,
                           'searchWindowSize': 10, 'h': 7, 'color': 21}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            self.options['src'] = np.array(self.options['src'])
       #     return cv2.fastNlMeansDenoisingColored(self.options['src'], None, self.options['templateWindowSize'], self.options['searchWindowSize'], self.options['h'], self.options['color'])
            return cv2.fastNlMeansDenoisingColored(self.options['src'], None, self.options['h'], self.options['color'], self.options['templateWindowSize'], self.options['searchWindowSize'])

    class Enhance:
        default_options = {'src': None, 'coef': 2}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            src = self.options['src']
            self.options['src'] = Img.fromarray(
                src) if isinstance(src, np.ndarray) else src
            enhancer = ImageEnhance.Contrast(self.options['src'])
            return enhancer.enhance(self.options['coef'])

    class Dilate:
        default_options = {'src': None, 'ksize': (1, 1), 'iterations': 4}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            src = np.array(self.options['src'])
            self.options['src'] = src if isinstance(
                src, np.ndarray) else np.array(src)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_CROSS, self.options['ksize'])

            return Img.fromarray(cv2.dilate(self.options['src'], kernel, self.options['iterations']))

    class Erode:
        default_options = {'src': None, 'ksize': (1, 1), 'iterations': 1}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            src = np.array(self.options['src'])
            self.options['src'] = src if isinstance(
                src, np.ndarray) else np.array(src)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_CROSS, self.options['ksize'])

            return Img.fromarray(cv2.erode(self.options['src'], kernel, self.options['iterations']))

    class Bilateral:
        default_options = {'src': None, 'd': 9,
                           'sigmaColor': 20, 'sigmaSpace': 20}
        '''d -	Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
    sigmaColor -	Filter sigma in the color space. A larger value of the parameter means
                  that farther colors within the pixel neighborhood (see sigmaSpace) 
                  will be mixed together, resulting in larger areas of semi-equal color.
    sigmaSpace -	Filter sigma in the coordinate space. A larger value of the parameter means 
                  that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). 
                  When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace'''

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            src = self.options['src']
            self.options['src'] = src if isinstance(
                src, np.ndarray) else np.array(src)
            return cv2.bilateralFilter(src, self.options['d'], self.options['sigmaColor'], self.options['sigmaSpace'], BORDER_TRANSPARENT)

    class Contr_brigh:
        default_options = {'src': None, 'contrast': 0.421, 'brightness': 0.9}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = self.options['src']
            if isinstance(im, np.ndarray):
                im = Img.fromarray(im)
            br, contr = brightness_calc(im), contrast_calc(im)
            im = ImageEnhance.Brightness(im).enhance(
                self.options['brightness'] / (1 if br == 0 else br))
            # im_show('brightness', im)
            im = np.array(ImageEnhance.Contrast(im).enhance(
                self.options['contrast'] / (1 if contr == 0 else contr)))
            # im_show('contrast', im)
            return im

    class Adjust_brightness:
        default_options = {'src': None, 'brightness': 0.3}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = self.options['src']
            im = im if isinstance(im, np.ndarray) else np.array(im)
            im = tf.image.adjust_brightness(im, self.options['brightness'])
            return im

    class Adjust_contrast:
        default_options = {'src': None, 'contrast': 3.}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = self.options['src']
            im = im if isinstance(im, np.ndarray) else np.array(im)
            im = tf.image.adjust_contrast(im, self.options['contrast'])
            return im

    class Resize:
        default_options = {'src': None, 'scale_percent': 200}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            image = np.array(self.options['src'])
            width = int(image.shape[1] * self.options['scale_percent'] / 100)
            height = int(image.shape[0] * self.options['scale_percent'] / 100)
            dim = (width, height)
            im = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            return im

    class Rotate:
        default_options = {'src': None}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = np.array(self.options['src'])
            angle = self.getangle(im)
            if not (angle == 0):
                im = Img.fromarray(im)
                im = im.rotate(angle)
            return im

        def getangle(self, im):
            ''' looking for contours, calculates its angle to horizont
            and retuns result'''

            img = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_REFLECT)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            erode_img = cv2.erode(blur_gray, kernel, iterations=8)
            # im_show('erode_img', erode_img)
            contours, hierarchy = cv2.findContours(
                erode_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            i, angle = 0, 0
            for c in contours:
                if hierarchy[0][i][3] == 0:
                    xy, wh, angle = cv2.minAreaRect(c)
                    break
                i += 1

            if angle > 0:
                return angle - 90 if angle > 60 else angle
            else:
                return angle + 90 if angle < -60 else angle

    class Background:
        default_options = {'src': None}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = np.array(self.options['src'])
            backSub = cv2.createBackgroundSubtractorMOG2()
            im = backSub.apply(np.float32(im))
            return im

    class Negative:
        default_options = {'src': None}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            im = np.array(self.options['src'])
            im = cv2.bitwise_not(im)
            return im

    class Black_white:
        default_options = {'src': None, 'thresh': 117}

        def __init__(self, **kwargs):
            self.options = self.default_options
            self.options.update(kwargs)

        def do(self):
            filter_image = self.options['src']
            im = Img.fromarray(filter_image) if isinstance(
                filter_image, np.ndarray) else filter_image

            def fn(x): return 255 if x > self.options['thresh'] else 0
            im = im.convert('L').point(fn, mode='1')
            return im


# ----------------- Filter instance-----------------------------
class Tuner_Instances:
    def __init__(self, path=''):
        self.filters = []
        self.images = []
        if len(path) > 0:
            self = self.load(path)
        else:
            self.filters = []
            self.images = []

    def construct(self, filter):
        filterClass = getattr(Tuner, filter['name'])
        instance = filterClass(**filter['param'])
        self.filters.append(instance)

    def filter_exec(self, image):
        self.images.append(('Init', np.array(image)))
        for tuning in self.filters:
            tuning.options['src'] = image
            image = tuning.do()
            self.images.append((str(tuning.__class__).split('.')[2], image))
        return image

    def plot(self):
        for i, im in enumerate(self.images):
            plt_show(im[1], im[0], i+1)

    def save(self, path='/Tuner_Instance'):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, path='/Tuner_Instance'):
        with open(path, 'rb') as file:
            return pickle.load(file)
