import pytest
import numpy as np
import dlib
from typing import List
from PIL import Image
from dlib_wrapper import dlibModelsLoader, dlibFaceProcessor
import os

### Test dlibModelsLoader ###
@pytest.fixture(scope="class")
def dlib_loader_instance():
    return dlibModelsLoader()

def test_directory_creation(dlib_loader_instance):
    assert os.path.exists(dlib_loader_instance.MODELS_DIR)

def test_download_unzip_shape_predictor(dlib_loader_instance):
    shape_predictor_zip = os.path.join(dlib_loader_instance.MODELS_DIR, "shape_predictor_5_face_landmarks.dat.bz2")
    shape_predictor = os.path.join(dlib_loader_instance.MODELS_DIR, "shape_predictor_5_face_landmarks.dat")

    dlib_loader_instance._download_file(dlib_loader_instance.SHAPE_PREDICTOR_URL, shape_predictor_zip)
    dlib_loader_instance._unzip_bz2_file(shape_predictor_zip, shape_predictor)

    assert os.path.isfile(shape_predictor_zip)
    assert os.path.isfile(shape_predictor)

def test_download_unzip_face_recognition(dlib_loader_instance):
    face_recognition_zip = os.path.join(dlib_loader_instance.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat.bz2")
    face_recognition = os.path.join(dlib_loader_instance.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

    dlib_loader_instance._download_file(dlib_loader_instance.FACE_RECOGNITION_URL, face_recognition_zip)
    dlib_loader_instance._unzip_bz2_file(face_recognition_zip, face_recognition)

    assert os.path.isfile(face_recognition_zip)
    assert os.path.isfile(face_recognition)

def test_load_dlib_models(dlib_loader_instance):
    shape_predictor = os.path.join(dlib_loader_instance.MODELS_DIR, "shape_predictor_5_face_landmarks.dat")
    face_recognition = os.path.join(dlib_loader_instance.MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor_path, face_recognition_path = dlib_loader_instance.load_dlib_models()

    assert shape_predictor_path == shape_predictor
    assert face_recognition_path == face_recognition


### Tests dlibFaceProcessor ###
@pytest.fixture(scope='class')
def face_processor():
    return dlibFaceProcessor()

@pytest.fixture
def sample_image():
    img = Image.open("./data/pulp-fiction.jpg")
    return np.array(img)

def test_detect_faces(face_processor, sample_image):
    boxes = face_processor.detect_faces(sample_image)

    assert isinstance(boxes, dlib.rectangles)
    assert all(isinstance(box, dlib.rectangle) for box in boxes)
    assert len(boxes) == 2

def test_get_shapes(face_processor, sample_image):
    boxes = face_processor.detect_faces(sample_image)
    shapes = face_processor.get_shapes(sample_image, boxes)

    assert isinstance(shapes, List)
    assert all(isinstance(shape, dlib.full_object_detection) for shape in shapes)

def test_align_faces(face_processor, sample_image):
    boxes = face_processor.detect_faces(sample_image)
    shapes = face_processor.get_shapes(sample_image, boxes)
    aligned_faces = face_processor.align_faces(sample_image, shapes)

    assert isinstance(aligned_faces, List)
    assert all(isinstance(face, np.ndarray) for face in aligned_faces)

def test_encode_faces(face_processor, sample_image):
    boxes = face_processor.detect_faces(sample_image)
    shapes = face_processor.get_shapes(sample_image, boxes)
    aligned_faces = face_processor.align_faces(sample_image, shapes)
    encodings = face_processor.encode_faces(aligned_faces)

    assert isinstance(encodings, List)
    assert all(isinstance(encoding, np.ndarray) for encoding in encodings)

def test_detect_and_encode_faces(face_processor, sample_image):
    encodings = face_processor.detect_and_encode_faces(sample_image)

    assert isinstance(encodings, List)
    assert all(isinstance(encoding, np.ndarray) for encoding in encodings)
