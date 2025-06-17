import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import glob
from basicsr.utils import imwrite
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# functions -------------------------------------------------------------------
class Facenet():
    net = cv.dnn.readNetFromCaffe(
        'experiments/facenet/deploy.prototxt.txt',
        'experiments/facenet/res10_300x300_ssd_iter_140000.caffemodel'
    )

    def __init__(self, 
                 input='emergency\\test', 
                 output='facenet_results',
                 database='emergency/database',
                 facenet_model_path = 'experiments/facenet/20180402-114759.pb'):
        self.input = input
        self.output = output
        self.facenet_model_path = facenet_model_path

        self.database = database
        self.known_embeddings = None
        self.known_labels = None
        self.svm_model = None

        self.facenet_model = self.load_facenet_model(facenet_model_path)
        os.sep = '/'
    
    # FACENET MODEL
    def load_facenet_model(self, model_path):
        facenet_model = tf.Graph()
        with facenet_model.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        print("facenet model is loaded")
        return facenet_model

    # FACE DETECTION MODELS
    def detect_faces_dnn(self, image, net, conf_threshold=0.3):
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
        
        print(f"Number of faces detected: {len(faces)}")
        return faces

    # MTCNN
    def detect_faces_mtcnn(self, image, conf_threshold=0.9, mode="multiple"):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = MTCNN().detect_faces(rgb_image)
        
        faces = []
        confidences = []
        for result in results:
            confidence = result['confidence']
            x, y, width, height = result['box']
            startX, startY, endX, endY = x, y, x + width, y + height
            
            if confidence >= conf_threshold:
                face = image[startY:endY, startX:endX]
                faces.append((startX, startY, endX, endY, face))
                confidences.append(confidence)
        
        if mode=="multiple":
            return faces
        elif mode=="single":
            return faces[np.argmax(confidences)]

    # return a preprocessed face (returns only one)
    def preprocess_face(self, face, image_size=160):
        face = cv.resize(face, (image_size, image_size))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        return face

    # return list of embeddings
    def generate_embeddings(self, facenet_model, faces):
        with facenet_model.as_default():
            with tf.compat.v1.Session(graph=facenet_model) as sess:
                images_placeholder = facenet_model.get_tensor_by_name("input:0")
                embeddings = facenet_model.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = facenet_model.get_tensor_by_name("phase_train:0")

                feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
                embeddings = sess.run(embeddings, feed_dict=feed_dict)
        
        print("embedding complete.")
        return embeddings

    def recognize_faces_svm(self, embeddings):
        recognized_faces = []
        if self.svm_model:
            predictions = self.svm_model.predict(embeddings)
            probabilities = self.svm_model.predict_proba(embeddings)
            
            for i, prediction in enumerate(predictions):
                confidence = max(probabilities[i])  # Get highest probability
                label = self.label_encoder.inverse_transform([prediction])[0] if confidence >= 0.1 else "Unknown"
                recognized_faces.append((label, confidence))
        
        return recognized_faces
    
    def process_database(self):
        database_path = self.database
        self.known_labels = []
        known_images = []
        
        if database_path.endswith('/'):
            database_path = database_path[:-1]
        if os.path.isfile(database_path):
            database_list = [database_path]
        else:
            database_list = sorted(glob.glob(os.path.join(database_path, '*')))

        for db_path in database_list:
            db_path = db_path.replace("\\","/")
            print(f"Before method call, img_path: {db_path} (type: {type(db_path)})")
            img_name = os.path.basename(db_path)
            basename, ext = os.path.splitext(img_name)

            db_image = cv.imread(db_path)
            db_image = self.detect_faces_mtcnn(db_image, mode="single")
            known_images.append(db_image)
            self.known_labels.append(basename)

        known_faces = known_images
        preprocessed_known_faces = [self.preprocess_face(face) for _, _, _, _, face in known_faces]
        self.known_embeddings = self.generate_embeddings(self.facenet_model, preprocessed_known_faces)

        # Encode labels and train SVM
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(self.known_labels)
        self.svm_model = SVC(kernel='linear', probability=True)
        self.svm_model.fit(self.known_embeddings, encoded_labels)
        print("SVM model trained.")

    def run_recognition(self):
        self.process_database()
        
        if self.input.endswith('/') or self.input.endswith('\\'):
            self.input = self.input[:-1]
        if os.path.isfile(self.input):
            test_list = [self.input]
        else:
            test_list = sorted(glob.glob(os.path.join(self.input, '*')))

        for test_path in test_list:
            print(test_path)
            img_name = os.path.basename(test_path)
            basename, ext = os.path.splitext(img_name)

            image = cv.imread(test_path)
            faces = self.detect_faces_mtcnn(image)

            if len(faces) > 0:
                preprocessed_faces = [self.preprocess_face(face) for _, _, _, _, face in faces]
                embeddings = self.generate_embeddings(self.facenet_model, preprocessed_faces)

                # Please append the code for AVM here, you may use functions.
                recognized_faces = self.recognize_faces_svm(embeddings)

                for (startX, startY, endX, endY, _), (label, distance) in zip(faces, recognized_faces):
                    cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 4)
                    cv.putText(image, f'{label} - {round(distance*100, 2)}%', (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
                
                imwrite(image, f'{self.output}/{basename}_svmrecognition{ext}')
            else:
                print("no faces found. repeating iteration.")
                imwrite(image, f'{self.output}/{basename}_svmrecognition{ext}')


# main -------------------------------------------------------------------
def main():
    FN = Facenet(input='captures', output='output', database='database')
    FN.run_recognition()

if __name__ == '__main__':
    main()

