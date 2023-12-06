from dlib_wrapper import dlibFaceProcessor
from PIL import Image
import numpy as np

# Load image and convert to np.ndarray
img = Image.open("./data/pulp-fiction.jpg")
img = np.array(img)

# Instantiate face processor
face_processor = dlibFaceProcessor()

# Image -> Embeddings (option 1)
# detect_and_encode_faces combines the 4 steps into 1
embeddings1 = face_processor.detect_and_encode_faces(img)

# Image -> Embeddings (option 2)
# 1. Detect faces, get face boxes
boxes = face_processor.detect_faces(img)
# 2. Extract face landmarks
shapes = face_processor.get_shapes(img, boxes)
# 3. Align faces
aligned_faces = face_processor.align_faces(img, shapes)
# 4. Extract face embeddings
embeddings2 = face_processor.encode_faces(aligned_faces)

# Resulting embeddings1 and embeddings2 are identical
