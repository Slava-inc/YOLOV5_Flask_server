from flask import Flask, request
# import cv2
# from matplotlib import image

# import numpy as np
# import json
import dataloader
import detect as yolo
# import img_to_txt as totext
import parameterPrediction as pp

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    image = dataloader.binary_load(request.data)
    # totext.im_show('main', image)
    '''
    # image pretuning
    filter_list = ['contr_brigh', 'bilateral', 'enhance', 'dilate', 'erode']

    tunable = totext.Tunable_Filter(image, filter_list, show=True)
    for op in filter_list:
        print(op)
        tunable.do_operation()
    image = tunable.filter_image
    '''
    # image = totext.pillow_adj(image, 0.421, 0.9)
    # image = totext.enhance(image)

    img_results = yolo.detect(image, False)

    if len(img_results) == 0:
        return ''
    # get all fields at once
    # return pp.images_to_text(img_results)
    result = pp.bd_to_text(img_results)
    return result

    '''json_res = {}
    for key in ['birth date', 'first name', 'second name', 'third name']:
        cur_img = img_results.get(key)
        if type(cur_img) == np.ndarray:
            digits = (key=='birth date')
            show = False  # (key == 'second name')
            json_res[key] = yolo.tune_get_txt(cur_img, digits=digits,
            lang= 'eng' if digits else 'rus', show=show)
        else:
            json_res[key] = ''
            return str(json_res)
    # return jsonify({'result': json_res})'''

    @app.route('/test', methods=['POST'])
    def test():
        img_results = dataloader.load_binary_image(request.data)
        if len(img_results) == 0:
            return ''

        json_res = {}
        key = 'first name'
        digits = False
        show = False

        json_res[key] = yolo.tune_get_txt(img_results,
                                          digits=digits, lang='eng' if digits
                                          else 'rus', show=show)
        return str(json_res)


if __name__ == '__main__':
    app.run('localhost', 5000, debug=True)
