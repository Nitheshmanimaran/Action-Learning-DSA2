import subprocess
import pandas as pd
import cv2
import time
import torch
from gtts.tts import gTTS
from pydub import AudioSegment

global IMG_FREQUENCY, CONFIDENCE, count_margin
IMG_FREQUENCY = 30
CONFIDENCE = 0.8
count_margin = 50  # to define how long do we keep the object history list,
# the unit is the number of image
AudioSegment.converter = "C:/Users/Lingfen/AppData/Local/Programs/" \
                         "ffmpeg/bin/ffmpeg.exe"


def load_yolo_model():
    weights_path = r'../model_weights/best (8).pt'
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights_path,
                           force_reload=False, trust_repo=True)
    return model


def object_predict_image(model, image, h, w, count, CONFIDENCE, df,
                         obj_list, obj_history, count_history, with_audio):
    font_scale = 0.5
    thickness = 2

    h_min, h_max = int(h * 0.2), int(h)
    w_min, w_max = int(w * 0.35), int(w * 0.65)

    ori_img = image.copy()
    temp_img_path = '../output/org_img.jpg'
    cv2.imwrite(temp_img_path, image)
    cv2.rectangle(image, (w_min, h_min), (w_max, h_max),
                  color=(0, 255, 0), thickness=thickness)

    if (count % IMG_FREQUENCY) == 0:
        start = time.perf_counter()
        # cv2.imwrite(f'../output/org_img.jpg', image)
        results = model(temp_img_path)
        df = results.pandas().xyxy[0]
        time_took = time.perf_counter() - start
        print(f'time_took: {time_took} seconds')

        # filter out the obj_list
        for _, row in df.iterrows():
            if row['confidence'] > CONFIDENCE:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), \
                                 int(row['xmax']), int(row['ymax'])
                if ((w_min < x1 < w_max) or (
                        w_min < x2 < w_max)) and (y2 > h_min):
                    obj_list.append(row['class'])

        obj_list = list(set(obj_list))
        # if the new object list is not in the history list,
        # translate into text
        for obj in obj_list:
            if obj not in obj_history:

                # translate into audio with gTTS
                if with_audio:
                    print('tts predicted', obj)
                    silence = AudioSegment.silent(duration=0.02 * 1000)
                    description = class_dic[obj] + ' in front!'
                    tts = gTTS(description, lang='en')
                    tts.save('../output/tts.mp3')
                    audio = AudioSegment.from_mp3('../output/audio.mp3')
                    tts_audio = AudioSegment.from_mp3('../output/tts.mp3')
                    audio = audio + tts_audio + silence
                    audio.export('../output/audio.mp3', format="mp3")

        # remove the oldest elements from history if it's older than the margin
        if len(count_history) != 0 and \
                (count - int(count_history[0][0]) > count_margin):
            obj_history = obj_history[int(count_history[0][1]):]
            count_history = count_history[1:]
        else:
            # otherwise add the element to the history list
            count_history.append([count, len(obj_list)])
            obj_history.extend(obj_list)

        print('The history obj list:', obj_history)
    # else:
    #     if with_audio==True:
    #         silence = AudioSegment.silent(duration=0.01 * 1000)
    #         audio = AudioSegment.from_mp3('../output/audio.mp3')
    #         audio = audio + silence
    #         audio.export('../output/audio.mp3', format="mp3")

    if not df.empty:
        obj_text = [class_dic[i] for i in obj_list]
        obj_text = str(obj_text)
        cv2.putText(image, obj_text, (w_min, h_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                    color=(0, 255, 0), thickness=thickness)

        for _, row in df.iterrows():
            if row['confidence'] > CONFIDENCE:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), \
                                 int(row['xmax']), int(row['ymax'])
                if ((w_min < x1 < w_max) or (
                        w_min < x2 < w_max)) and (y2 > h_min):
                    cv2.rectangle(image, (x1, y1), (x2, y2),
                                  color=(0, 0, 255), thickness=thickness)
                    text = f"{class_dic[int(row['name'])]}: " \
                           f"{row['confidence']:.2f}"
                    # calculate text width & height to draw the transparent
                    # boxes as background of the text
                    (text_width, text_height) = \
                        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x1
                    text_offset_y = y1 - 5
                    box_coords = ((text_offset_x, text_offset_y),
                                  (text_offset_x + text_width + 2,
                                   text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1],
                                  color=(100, 100, 0), thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    # now put the text (label: confidence %)
                    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale, color=(0, 0, 255),
                                thickness=thickness)
        return obj_list, image, df, obj_history, count_history

    else:
        return [], ori_img, pd.DataFrame(), obj_history, count_history


def object_detection_video(yolo_model, video_path, output_result_text,
                           confidence=0.5, with_audio=False):
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    h, w = image.shape[:2]

    count = 0
    img_array = []
    df = pd.DataFrame()
    obj_list = []
    obj_history = []
    count_history = []
    if with_audio:
        silence = AudioSegment.silent(duration=0.01 * 1000)
        silence.export('../output/audio.mp3', format="mp3")

    while True:  # cap.isOpen():
        ret, image = cap.read()
        if ret:
            if with_audio:
                obj_list, pred_image, df, obj_history, count_history = \
                    object_predict_image(yolo_model, image, h, w, count,
                                         CONFIDENCE, df, obj_list, obj_history,
                                         count_history, True)
            else:
                obj_list, pred_image, df, obj_history, count_history = \
                    object_predict_image(yolo_model, image, h, w, count,
                                         CONFIDENCE, df, obj_list, obj_history,
                                         count_history, False)
            img_array.append(pred_image)
        else:
            break
        count += 1

    fource = cv2.VideoWriter_fourcc(*'mpv4')
    result_path = '../output/predicted_video.mp4'
    out = cv2.VideoWriter('../output/predicted.mp4', fource, 30, (w, h))
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    if with_audio:
        cmd = 'ffmpeg -i ../output/predicted.mp4 -i ../output/audio.mp3 ' \
              '-c copy ../output/predicted_video.mp4'
        subprocess.call(cmd, shell=True)
    else:
        result_path = '../output/predicted.mp4'

    with open(output_result_text, 'w') as f:
        f.write(result_path)

    cap.release()
    cv2.destroyAllWindows()

    return result_path


def get_classes_dict(csv_path):
    df = pd.read_csv(csv_path)
    class_dic = df['name'].to_dict()
    return class_dic


csv_path = '../input/classes.csv'
class_dic = get_classes_dict(csv_path)
