import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import glob
from basicsr.utils import imwrite
from mtcnn import MTCNN


# functions -------------------------------------------------------------------
def load_facenet_model(model_path):
    facenet_model = tf.Graph()
    with facenet_model.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    print("facenet model is loaded")
    return facenet_model

net = cv.dnn.readNetFromCaffe(
    'experiments/pretrained_models/facenet/deploy.prototxt.txt',
    'experiments/pretrained_models/facenet/res10_300x300_ssd_iter_140000.caffemodel'
)

facenet_model_path = 'experiments/pretrained_models/facenet/20180402-114759.pb'
facenet_model = load_facenet_model(facenet_model_path)

# return list of cropped faces
# OpenCV_DNN
def detect_faces_dnn(image, net, conf_threshold=0.3):
    h,w = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            faces.append((startX, startY, endX, endY, face))

            h1, w1 = face.shape[:2]
            print(startX, startY, endX, endY)
            print(f"height: {h1} width: {w1}")
    
    print(f"Number of faces detected: {len(faces)}")
    return faces

# MTCNN
detector = MTCNN()
def detect_faces_mtcnn(image, conf_threshold=0.9):
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_image)
    
    faces = []
    for result in results:
        confidence = result['confidence']
        x, y, width, height = result['box']
        startX, startY, endX, endY = x, y, x + width, y + height
        
        if confidence >= conf_threshold:
            face = image[startY:endY, startX:endX]
            faces.append((startX, startY, endX, endY, face))
    
    return faces

# return a preprocessed face (returns only one)
def preprocess_face(face, image_size=160):
    face = cv.resize(face, (image_size, image_size))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return face

# return list of embeddings
def generate_embeddings(facenet_model, faces):
    with facenet_model.as_default():
        with tf.compat.v1.Session(graph=facenet_model) as sess:
            images_placeholder = facenet_model.get_tensor_by_name("input:0")
            embeddings = facenet_model.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = facenet_model.get_tensor_by_name("phase_train:0")

            feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
            embeddings = sess.run(embeddings, feed_dict=feed_dict)
    
    print("embedding complete.")
    return embeddings

from scipy.spatial.distance import cosine
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)  # Similarity score (0 to 1)

def recognize_faces(known_embeddings, known_labels, embeddings, threshold=0.5):
    recognized_faces = []
    for embedding in embeddings:
        distances = []
        for known_embedding in known_embeddings:
            distance = 1 - cosine(embedding, known_embedding)
            distances.append(distance)
        
        best_distance = np.max(distances)
        print(distances)
        if best_distance >= threshold:
            index = np.argmax(distances)
            recognized_faces.append(known_labels[index])
        else:
            recognized_faces.append("Unknown")
    
    return recognized_faces


# database -------------------------------------------------------------------
database_path = "emergency/database"
known_images = []
known_labels = []

if database_path.endswith('/'):
    database_path = database_path[:-1]
if os.path.isfile(database_path):
    database_list = [database_path]
else:
    database_list = sorted(glob.glob(os.path.join(database_path, '*')))

for img_path in database_list:
    img_name = os.path.basename(img_path)
    basename, ext = os.path.splitext(img_name)
    known_images.append(img_path)
    known_labels.append(basename)

known_faces = [cv.imread(img) for img in known_images]
preprocessed_known_faces = [preprocess_face(face) for face in known_faces]
known_embeddings = generate_embeddings(facenet_model, preprocessed_known_faces)


# main -------------------------------------------------------------------
# test_path = 'emergency\\test'
test_path = 'emergency\\test_HR'
if test_path.endswith('/'):
    test_path = test_path[:-1]
if os.path.isfile(test_path):
    test_list = [test_path]
else:
    test_list = sorted(glob.glob(os.path.join(test_path, '*')))

print(test_list)

for test_path in test_list:
    print(test_path)
    img_name = os.path.basename(test_path)
    basename, ext = os.path.splitext(img_name)

    image = cv.imread(test_path)
    faces = detect_faces_mtcnn(image)

    if len(faces) > 0:
        preprocessed_faces = [preprocess_face(face) for _, _, _, _, face in faces]
        embeddings = generate_embeddings(facenet_model, preprocessed_faces)

        recognized_faces = recognize_faces(known_embeddings, known_labels, embeddings)

        for (startX, startY, endX, endY, _), label in zip(faces, recognized_faces):
            cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 4)
            cv.putText(image, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
        
        imwrite(image, f'facenet_results/{basename}_recognition{ext}')
    else:
        print("no faces found. repeating iteration.")
        imwrite(image, f'facenet_results/{basename}_HR_recognition{ext}')
