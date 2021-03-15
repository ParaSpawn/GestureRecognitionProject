import torchvision.transforms as transforms
import torch as T
import cv2
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label
import time
import Nets.GenericNet as GN
from Nets.GenericNet import WEIGHT_PATHS


url = 'http://192.168.0.101:8080/video'

cap = cv2.VideoCapture(0)

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

pil_to_tensor = transforms.ToTensor()

hand_detector = GN.GenericNet(GN.PATHS['HANDDETECTOR'], device).to(device)
hand_detector.load_state_dict(T.load(WEIGHT_PATHS['HANDDETECTOR']))
feature_estimator = GN.GenericNet(GN.PATHS['POSEMACHINE'], device).to(device)
feature_estimator.load_state_dict(T.load(WEIGHT_PATHS['POSEMACHINE']))

def drawBoundingBox(imgcv, result):
    for box in result:
        # print(box)
        x1, y1, x2, y2 = (box['topleft']['x'], box['topleft']['y'],
                          box['bottomright']['x'], box['bottomright']['y'])
        conf = box['confidence']
        # print(conf)
        label = box['label']
        if conf < self.predictThresh:
            continue
        # print(x1,y1,x2,y2,conf,label)
        cv2.rectangle(imgcv, (x1, y1), (x2, y2), (0, 255, 0), 6)
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
        # print('labelSize>>',labelSize)
        _x1 = x1
        _y1 = y1  # +int(labelSize[0][1]/2)
        _x2 = _x1+labelSize[0][0]
        _y2 = y1-int(labelSize[0][1])
        cv2.rectangle(imgcv, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgcv, label, (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    return imgcv

def get_point(preds):
    max_pos = None
    m = 0
    for i in range(preds.size(2)):
        for j in range(preds.size(3)):
            if preds[0,0,i,j] > m:
                m = preds[0,0,i,j]
                max_pos = (i, j)
    return max_pos

def get_all_peaks(heatmap_avg):
    all_peaks = []
    thre = 0.3
    for part in range(5):
        map_ori = heatmap_avg[0, part, :, :].detach().cpu()
        one_heatmap = gaussian_filter(map_ori, sigma=3)
        binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
        if np.sum(binary) == 0:
            all_peaks.append([0, 0])
            continue
        label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
        max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
        label_img[label_img != max_index] = 0
        map_ori[label_img == 0] = 0

        y, x = util.npmax(map_ori)
        all_peaks.append([x, y])
    return np.array(all_peaks)

fps = 0
lt = time.time()

hand_detector_lt = time.time()
hand_pose_lt = time.time()
feature_estimator(T.zeros(1, 3, 200, 200).to(device))

pos = None
while(True):
    ret, frame = cap.read()
    img_tensor = T.FloatTensor(frame).to(device).view(3, frame.shape[0], frame.shape[1])
    if time.time() - hand_detector_lt > 1 / 10:
        preds = hand_detector(img_tensor.unsqueeze(0))[-1]
        pos = get_point(preds)
        pos = (int((pos[0] / preds.size(2)) * img_tensor.size(1)), int((pos[1] / preds.size(3)) * img_tensor.size(2)))
        hand_detector_lt = time.time()
    if time.time() - hand_pose_lt > 1 / 6:
        preds = feature_estimator(img_tensor.unsqueeze(0))[-1]
        all_peaks = get_all_peaks(preds)
    cv2.circle(frame, pos, 5, (1, 0, 0), -1)
    cv2.imshow('Video', frame)
    fps += 1
    if time.time() - lt > 1:
        lt = time.time()
        print(fps)
        fps = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
