import numpy as np
import os
from glob import glob
import re
import detect
import cv2
import pickle
import dataloader

path = '//buchsrv00/f$/personal data/Logist_2022'

folders = []
files_list = []
for i, (d, dir, files) in enumerate(os.walk(path)):
    folders.append((d, files))
    for file in list(glob(os.path.join(d, '*.pdf'))):
        if re.findall(r'[П, п]аспорт', file):
            file_path = re.sub(r'\\', '/', file)
            with open(file_path, 'rb') as image:
                img = dataloader.binary_load(image.read())
            img_results = detect.detect(img, False)
            if len(img_results) == 4:
                files_list.append((file_path, img_results))
with open('id_main_Logist.pkl', 'wb') as file:
    pickle.dump(files_list, file)
