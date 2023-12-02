from typing import List
import numpy as np
import dlib
import bz2
import gdown
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def load_image_from_web_link(image_path: str) -> np.ndarray:

        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))

        if (".png" in image_path) or (".PNG" in image_path):
            jpeg_image_io = BytesIO()
            img.convert('RGB').save(jpeg_image_io, format='JPEG')
            jpeg_image_io.seek(0)
            img = Image.open(jpeg_image_io)
            print('Converted a png image to jpg.')

        return np.array(img)


def unzip_bz2_file(zipped_file_name):
    zipfile = bz2.BZ2File(zipped_file_name)
    data = zipfile.read()
    newfilepath = zipped_file_name[:-4] #discard .bz2 extension
    open(newfilepath, 'wb').write(data)

def download_file(url):
    output = "./dlib_models/" + url.split("/")[-1]
    print(output)
    gdown.download(url, output, quiet=False)



filepath_shape_predictor = "./dlib_models/shape_predictor_5_face_landmarks.dat.bz2"
filepath_face_recognition = "./dlib_models/dlib_face_recognition_resnet_model_v1.dat.bz2"

if os.path.isfile('./dlib_models/shape_predictor_5_face_landmarks.dat') != True:
    print("shape_predictor_5_face_landmarks.dat is going to be downloaded")
    url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
    download_file(url)
    unzip_bz2_file(filepath_shape_predictor)

if os.path.isfile('./dlib_models/dlib_face_recognition_resnet_model_v1.dat') != True:
    print("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")
    url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    download_file(url)
    unzip_bz2_file(filepath_face_recognition)

class dlibWrapper():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("./dlib_models/shape_predictor_5_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("./dlib_models/dlib_face_recognition_resnet_model_v1.dat")
        pass

    def face_detector(self, img: np.ndarray):
        boxes = self.detector(img, 1)  # 1 is the upsampling factor, the bigger the more computation
        return boxes

    def shape_predictor(self, img: np.ndarray, boxes):
        img_shapes = [self.sp(img, box) for box in boxes]
        return img_shapes

    def face_aligning(self, img: np.ndarray, img_shapes) -> List[np.ndarray]:
        aligned_faces = [dlib.get_face_chip(img, img_shape) for img_shape in img_shapes]
        return aligned_faces

    def face_encoder(self, aligned_faces) -> List[np.ndarray]:
        embeddings = [np.array(self.facerec.compute_face_descriptor(face)) for face in aligned_faces]
        return embeddings

    def detect_and_encode_faces(self, img: np.ndarray) -> List[np.ndarray]:
        boxes = self.detector(img, 1)
        img_shapes = [self.sp(img, box) for box in boxes]
        aligned_faces = [dlib.get_face_chip(img, img_shape) for img_shape in img_shapes]
        embeddings = [np.array(self.facerec.compute_face_descriptor(face)) for face in aligned_faces]
        return embeddings


if __name__=="__main__":
    img = load_image_from_web_link("https://res.cloudinary.com/chelsea-production/image/upload/c_fit,h_630,w_1200/v1/editorial/news/2022/01/15/Lewis-Baker-Chestefield-stock")
    dl = dlibWrapper()
    embeddings = dl.detect_and_encode_faces(img)
    print("The length: ", len(embeddings))