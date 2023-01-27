import pickle
import argparse
from pathlib import Path

import cv2
import torch 
import numpy as np
from dataloader import convert_pdf, letterbox

from models.experimental import attempt_load

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from plots import plot_one_box
from PIL import Image
from img_to_txt import get_txt, tune_get_txt


def detect(img0, save_img=False):

    # test
    # path = './kibitkin.jpg'
    # img0 = convert_pdf(path) if path.endswith('pdf') else cv2.imread(path)
    
    half = False
    device = 'cpu'
    txt_show = False

    source = Path('/pickle/images')
    imgsz = 416
    conf_thres = 0.25
    iou_thres = 0.45
    save_txt = False
    save_conf = True
    project = 'runs/detect'
    dataset = None
    # Directories
    save_dir = Path(increment_path(Path(project) / 'pickle', exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    with open('yolov5_ID3.pkl', 'rb') as pkl_file:
        model = pickle.loads(pickle.load(pkl_file))
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    ## Padded resize
    img = letterbox(img0, imgsz, stride)[0]
    ## Convert
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]

    # Apply NMS
    # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)
    txt_img = {}    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # p, s = path, ''
        s = ''
        
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results

            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # with open(txt_path + '.txt', 'a') as f:
                    #   f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=0)
                nm = names[int(cls)]
                txt_img[nm] = img0[int(xyxy[1] ):int(xyxy[3] * (1 if nm == 'birth date' else 1.004)), int(xyxy[0] * 0.9):int(xyxy[2] * 1.1)] # int(xyxy[0]*0.9):int(xyxy[2]*1.1)
                tmp_img = Image.fromarray(txt_img[nm])
                name_tmp = nm + '.jpeg'
                tmp_img.save(name_tmp, dpi=(300, 300))
                txt_img[nm] = np.array(Image.open(name_tmp))
                if txt_show:

                    cv2.imshow(nm, txt_img[nm])
                    cv2.waitKey(100)
                    # Stream results
                    # cv2.imshow(str(p), img0)
                    # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #    cv2.imwrite(save_path, img0)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    return txt_img


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    # path = './228325116_7615c4467f7c40297760b86a135d5539_800_jpeg.rf.db38652437440090e56e3846a3e1e23e.jpg'
    # path = './Kibitkin_ID.pdf'
    path = './kibitkin.jpg'

    image = convert_pdf(path) if path.endswith('pdf') else cv2.imread(path)

    assert image is not None, 'Image Not Found ' + path
    img_results = detect(image, False)

    json_res = {}
    json_res['birth date'] = tune_get_txt(img_results['birth date'], show=False, digits=True)
    json_res['first name'] = tune_get_txt(img_results['first name'], 'rus')
    json_res['second name'] = tune_get_txt(img_results['second name'], 'rus')
    json_res['third name'] = tune_get_txt(img_results['third name'], 'rus')
    print('{} \n {} \n {} \n  {} \n'.format(json_res['birth date'], json_res.get('first name'), json_res.get('second name'), json_res.get('third name')))
'''    