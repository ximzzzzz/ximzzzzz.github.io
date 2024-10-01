---
title: FastAPI를 통한 기본 서빙
category: ML
tags: [ML, AI, serving, fastapi]
toc: true
math: true 
---

# FastAPI 를 통한 기본 서빙

포스팅의 설명과 코드는 coursera 강좌 [Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) 을 참고하여 작성했습니다

- FastAPI 를 활용하여 모델이 어떻게 서빙되는지 이해하기 위해 작성했습니다.
- 포스팅과 코드의 목적은 서빙이므로 모델은 라이브러리를 통해 기본적인 yolo3를 사용합니다.
- 서버는 이미지를 입력으로 받아 yolo3 를 통해 객체를 탐지하고 바운딩박스와 confidence를 표시하여 제공합니다. 

## 왜 FastAPI 인가?
- 완벽한 웹애플리케이션이나 보일러플레이트의 작성없이 ML모델 추론용 웹서버를 손쉽게 구축할 수 있습니다.
- 서버와 상호작용 할 수있는 빌트인 클라이언트를 제공합니다 

## 코드 파헤치기 

1. fastapi 인스턴스를 생성한다.
   - import 문 바로 다음에 위치시킴으로써 모든 라우터 및 미들웨어가 이 인스턴스를 참조할 수 있습니다

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# Create FastAPI instance
app = FastAPI(title='Deploying an ML Model with FastAPI')

```

2. 서빙에 필요한 여러가지 함수 정의
- 서빙에 사용할 입력데이터 유효성검사, 저장하기 등의 함수를 정의한다

```python
import os
import io
from enum import Enum

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException


class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


def create_output_dir(dir_name="images_uploaded"):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def validate_image(filename: str) -> bool:
    return filename.split(".")[-1] in ("jpg", "jpeg", "png")


def read_image_file(file: UploadFile) -> np.ndarray:
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# Run object detection
def detect_objects(image: np.ndarray, model: Model) -> tuple:
    return cv.detect_common_objects(image, model=model)


def create_output_image(image: np.ndarray, bbox: list, label: list, conf: list) -> np.ndarray:
    return draw_bbox(image, bbox, label, conf)


def save_image(image: np.ndarray, filename: str) -> None:
    cv2.imwrite(f'images_uploaded/{filename}', image)
```

3. 서빙 코드 작성

```python
# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://docs"


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    if not validate_image(file.filename):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    image = read_image_file(file)

    # 3. RUN OBJECT DETECTION MODEL
    bbox, label, conf = detect_objects(image, model)

    # Create image that includes bounding boxes and labels
    output_image = create_output_image(image, bbox, label, conf)

    # Save it in a folder within the server
    save_image(output_image, file.filename)

    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    file_image = open(f'images_uploaded/{file.filename}', mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")
```

4. app 인스턴스를 유비콘을 통해 비동기화상태로 실행

```python
import uvicorn


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)

```
