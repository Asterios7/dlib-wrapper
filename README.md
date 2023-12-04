![Tests](https://github.com/Asterios7/dlib-wrapper/actions/workflows/tests.yaml/badge.svg)

# dlib-wrapper

The `dlib-wrapper` package provides a convenient interface for leveraging Dlib's facial processing capabilities in Python applications.

The `dlibFaceProcessor` class is the core component of this package, designed to streamline face detection, alignment, and encoding using the Dlib library in Python.

[**Installation**](#installation)
| [**Usage**](#usage)

## Installation<a id="installation"></a>

You can install the package directly from the GitHub repository using pip:

`pip install git+https://github.com/Asterios7/dlib-wrapper.git`

## Usage<a id="usage"></a>

Import the dlibFaceProcessor class into your Python code:

```python
from dlib_wrapper import dlibFaceProcessor
```

Initialize an instance of the dlibFaceProcessor class to access its range of functionalities:

```python
face_processor = dlibFaceProcessor()
```

#### Methods

`detect_faces(img: np.ndarray) -> dlib.rectangles`
Identifies faces in an image using dlib.get_frontal_face_detector().

`get_shapes(img: np.ndarray, boxes: List[dlib.rectangle]) -> List[dlib.full_object_detection]`
Predicts facial landmarks using dlib.shape_predictor.

`align_faces(img: np.ndarray, shapes: List[dlib.full_object_detection]) -> List[np.ndarray]`
Aligns detected faces using dlib.get_face_chip.

`encode_faces(faces: List[np.ndarray]) -> List[np.ndarray]`
Encodes aligned faces using dlib.face_recognition_model_v1.

`detect_and_encode_faces(img: np.ndarray) -> List[np.ndarray]`
Pipeline to detect and encode faces from an image.

#### Example usage

Find detailed usage examples in the `examples` folder of this repository.
