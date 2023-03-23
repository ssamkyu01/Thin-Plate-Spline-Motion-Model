from typing import Union
from fastapi import FastAPI, File, UploadFile
import shutil
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
from vid2img import VideoToImage
from io import BytesIO
import PIL
from IPython.display import display, Image, HTML
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/result", StaticFiles(directory="result"), name="result")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.put("/api/upload")
async def uploadImage(file: UploadFile = File(...)):

    print(file.file)
    print(file.filename)

    with open("assets/" + file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename}

    


@app.put("/api/generator")
async def generateImage(source: str):
    
    if source:
        print("/api/generator:"+source)
        device = torch.device('cuda:0')
        dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
        source_image_path = './assets/' + source

        driving_video_path = './assets/driving.mp4'
        output_video_path = './result/generated.mp4'
        config_path = 'config/vox-256.yaml'
        checkpoint_path = 'checkpoints/vox.pth.tar'
        predict_mode = 'relative' # ['standard', 'relative', 'avd']
        find_best_frame = True # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

        pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
        if(dataset_name == 'ted'): # for ted, the resolution is 384*384
            pixel = 384

        source_image = imageio.imread(source_image_path)
        reader = imageio.get_reader(driving_video_path)


        source_image = resize(source_image, (pixel, pixel))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

        if predict_mode=='relative' and find_best_frame:
            from demo import find_best_frame as _find
            print(source_image, device.type)
            i = _find(source_image, driving_video, device.type=='cpu')
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

        #save resulting video
        imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        videoToImage = VideoToImage(savePath="./result", videoPath="./result/generated.mp4")
        savedImageList = videoToImage.saveImage()

        return savedImageList
    
    else:

        return []