import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance
# import numpy as np
# import os
# #import tensorflow as tf
# import tensorflow_hub as hub
# import time ,sys
# from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import torch


IMG_FREQUENCY = 20
CONFIDENCE = 0.5

def load_yolo_model():
    weights_path = r'../models/yolov7_training.pt'
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights_path,
                           force_reload=True, trust_repo=True)
    return model


def object_predict_image(model, image, h, w, count, CONFIDENCE, df, obj_list):
    font_scale = 0.5
    thickness = 1

    h_min, h_max = int(h * 0.2), int(h)
    w_min, w_max = int(w * 0.25), int(w * 0.75)

    ori_img = image.copy()
    cv2.imwrite(f'../output/org_img.jpg', image)
    if count % IMG_FREQUENCY==0:
        start = time.perf_counter()
        # cv2.imwrite(f'../output/org_img.jpg', image)
        results = model('../output/org_img.jpg')
        df = results.pandas().xyxy[0]
        time_took = time.perf_counter() - start
        print(f'time_took: {time_took} seconds', time_took)
        obj_list = list(set(df['class']))

    cv2.rectangle(image, (w_min, h_min), (w_max, h_max), color=(0, 255, 0), thickness=thickness)

    if type(df)!='NoneType':
        obj_text = str(obj_list)
        cv2.putText(image, obj_text, (w_min, h_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        for _, row in df.iterrows():
            if row['confidence'] > CONFIDENCE:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                if ((w_min < x1 < w_max) or (
                        w_min < x2 < w_max)) and (y2 > h_min):
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=thickness)
                    text = f"{row['name']}: {row['confidence']:.2f}"
                    # calculate text width & height to draw the transparent boxes as background of the text
                    (text_width, text_height) = \
                        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                        thickness=thickness)[
                            0]
                    text_offset_x = x1
                    text_offset_y = y1 - 5
                    box_coords = ((text_offset_x, text_offset_y),
                                  (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(100, 100, 0), thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    # now put the text (label: confidence %)
                    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        return obj_list, image, df

    else:
        return [], ori_img, None


def object_detection_video(yolo_model, sst_model, confidence = 0.5):

    vid = '../input_video.mp4'

    cap = cv2.VideoCapture(vid)
    ret, image = cap.read()
    h, w = image.shape[:2]
    # print(f'the width is {w} and height is {h}')

    # fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    # out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
    count = 0
    img_array = []
    df = None
    obj_list = []

    while True: # cap.isOpen():
        ret, image = cap.read()
        if ret==True:
            obj_list, pred_image, df = object_predict_image(yolo_model, image, h, w, count, CONFIDENCE, df, obj_list)
            img_array.append(pred_image)
            # print(obj_list)
            # cv2.imwrite('../output/predicted_img.jpg', image)
            # cv2.imshow('result', image)
            if len(obj_list)!=0:
                print('SST:', obj_list)
        else: break
        count+=1

    fource = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter('../output/predicted_video.mp4', fource, 15, (w, h))
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

        #         out.write(image)
        #         cv2.imshow("image", image)
        #
        #     st.write(obj_list)
        #
        #     # ====================SST model prediction=========================
        #     sst_model = 'to apply the sst prediction'
        #
        #
        #     if ord("q") == cv2.waitKey(1):
        #         break
        # else:
        #     break
        # count+=1

    # return "detected_video.mp4"

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    yolo_model = load_yolo_model()
    sst_model = True
    start = time.perf_counter()
    object_detection_video(yolo_model, sst_model, CONFIDENCE)
    time_took = time.perf_counter() - start
    print(f'The total time spent to process the video is: {time_took} seconds', time_took)

