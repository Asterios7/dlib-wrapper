from typing import List
import numpy as np
import dlib
import os
import bz2
import gdown
from typing import Tuple

class dlibModelsLoader:
    MODELS_DIR = "./models"
    SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
    FACE_RECOGNITION_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    
    def __init__(self):
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)
            print(f"Directory '{self.MODELS_DIR}' created successfully.")
    
    def _download_file(self, url: str, output_path: str):
        if not os.path.isfile(output_path):
            print(f"Downloading {url}...")
            gdown.download(url, output_path, quiet=False)
            print(f"Downloaded {url} to {output_path}")
        else:
            print(f"{output_path} already exists. Skipping download.")
    
    def _unzip_bz2_file(self, zipped_filepath: str, new_filepath: str):
        if not os.path.isfile(new_filepath):
            print(f"Unzipping {zipped_filepath}...")
            with bz2.BZ2File(zipped_filepath, 'rb') as zipfile:
                data = zipfile.read()
                with open(new_filepath, 'wb') as f:
                    f.write(data)
            print(f"Unzipped {zipped_filepath} to {new_filepath}")
        else:
            print(f"{new_filepath} already exists. Skipping unzip.")
    
    def load_dlib_models(self) -> Tuple[str]:
        shape_predictor_zip = os.path.join(self.MODELS_DIR, "shape_predictor_5_face_landmarks.dat.bz2")
        shape_predictor = os.path.join(self.MODELS_DIR, "shape_predictor_5_face_landmarks.dat")
        face_recognition_zip = os.path.join(self.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat.bz2")
        face_recognition = os.path.join(self.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
        
        self._download_file(self.SHAPE_PREDICTOR_URL, shape_predictor_zip)
        self._unzip_bz2_file(shape_predictor_zip, shape_predictor)
        
        self._download_file(self.FACE_RECOGNITION_URL, face_recognition_zip)
        self._unzip_bz2_file(face_recognition_zip, face_recognition)
        return shape_predictor, face_recognition


class dlibFaceProcessor:
    def __init__(self):
        models_loader = dlibModelsLoader()
        shape_predictor_path, face_recognition_path = models_loader.load_dlib_models()

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_recognition = dlib.face_recognition_model_v1(face_recognition_path)

    def detect_faces(self, img: np.ndarray) -> dlib.rectangles:
        """Uses dlib.get_frontal_face_detector() to identify faces in an image"""
        return self.detector(img, 1)

    def get_shapes(self, img: np.ndarray, boxes: List[dlib.rectangle]) -> List[dlib.full_object_detection]:
        """Predicts facial landmarks using dlib.shape_predictor"""
        return [self.shape_predictor(img, box) for box in boxes]

    def align_faces(self, img: np.ndarray, shapes: List[dlib.full_object_detection]) -> List[np.ndarray]:
        """Aligns detected faces using dlib.get_face_chip"""
        return [dlib.get_face_chip(img, shape) for shape in shapes]

    def encode_faces(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """encodes aligned faces using dlib.face_recognition_model_v1"""
        return [np.array(self.face_recognition.compute_face_descriptor(face)) for face in faces]

    def detect_and_encode_faces(self, img: np.ndarray) -> List[np.ndarray]:
        """Pipeline to detect and encode faces from image"""
        boxes = self.detect_faces(img)
        shapes = self.get_shapes(img, boxes)
        aligned_faces = self.align_faces(img, shapes)
        return self.encode_faces(aligned_faces)
