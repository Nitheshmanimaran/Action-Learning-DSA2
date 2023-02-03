
from fastapi import FastAPI, File, UploadFile
import uvicorn
from object_detection import *
from fastapi.responses import FileResponse
from io import BytesIO


app = FastAPI(title='Navigation assistance system', version='1.0',
              description='Yolov7 object detector and google TTS')

model = load_yolo_model()
csv_path = '../input/classes.csv'
class_dic = get_classes_dict(csv_path)


@app.post("/upload", status_code=201)
async def upload(file: UploadFile = File(...)):

    video_file = BytesIO(await file.read())

    # move the file pointer to the beginning of the BytesIO object
    video_file.seek(0)

    # write the contents of the BytesIO object to a new file on disk
    input_video_path = "../input/uploaded_video.mp4"
    with open(input_video_path, "wb") as f:
        f.write(video_file.read())

    output_result_text = '../output/results.txt'

    start = time.perf_counter()
    object_detection_video(model, input_video_path,
                           output_result_text, CONFIDENCE, True)
    time_took = time.perf_counter() - start
    print(f'The total time spent to process the video is: {time_took} seconds')

    return FileResponse('../output/results.txt')

    # return {"file_name":file.file}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
