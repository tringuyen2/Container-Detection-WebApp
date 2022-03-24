from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from .detect import get_yolov5, get_image_from_bytes, read_image
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

import uuid




from fastapi.templating import Jinja2Templates






model = get_yolov5()

app = FastAPI(
    title="Using YOLOV5 for Container Detection API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

app.mount("/temp", StaticFiles(directory="temp"), name="temp")

templates = Jinja2Templates(directory="Frontend")

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


@app.post("/object-to-json")
async def detect_container_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_container_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save('temp/result.png')
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


@app.post("/detect")
async def detect_container(image: UploadFile = File(...)):
    temp_file = _save_file_to_disk(image, path="temp", save_as="temp")
    input_image = read_image('temp/temp.png')
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        name = f"{str(uuid.uuid4())}.png"
        # cv2.imwrite(name, output)
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(f'temp/{name}')
        # img_base64.save('temp/result1.png')
        # img_base64.save(bytes_io, format="jpeg")
    # return {"filename": image.filename, "text": "Processing"}
    return {"name": name}





def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file


