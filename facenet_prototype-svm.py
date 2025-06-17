import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import glob
import pickle
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import pandas as pd
from datetime import datetime
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, './')

class Facenet():
    def __init__(self, 
                 input='captures/standard/aaron', 
                 output='captures_rec_cos/standard/aaron',
                 database='database',
                 facenet_model_path='experiments/facenetv11.pb'):
        self.input = input
        self.output = output
        self.facenet_model_path = facenet_model_path

        self.database = database
        self.known_embeddings = None
        self.known_labels = None
        self.svm_classifier = None
        self.label_encoder = None

        self.facenet_graph, self.input_tensor, self.output_tensor, self.phase_train_tensor = self.load_facenet_model(facenet_model_path)
        self.sess = tf.compat.v1.Session(graph=self.facenet_graph)
        os.sep = '/'
    
    # Load The Facenet Model
    def load_facenet_model(self, model_path):
        try:
            facenet_graph = tf.Graph()
            with facenet_graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(model_path, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
            print("Facenet model loaded")
            input_tensor = facenet_graph.get_tensor_by_name('input:0')
            output_tensor = facenet_graph.get_tensor_by_name('embeddings:0')
            phase_train_tensor = facenet_graph.get_tensor_by_name('phase_train:0')
        except Exception as e:
            print(f"Error loading model: {e}")
            facenet_graph = None
            input_tensor = None
            output_tensor = None
            phase_train_tensor = None
        return facenet_graph, input_tensor, output_tensor, phase_train_tensor

    # Face Detection
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

    # Preprocessing
    def preprocess_face(self, face, image_size=160):
        face = cv.resize(face, (image_size, image_size))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        return face

    # Extract embeddings
    def generate_embeddings(self, faces):
        if self.facenet_graph is None:
            print("Facenet model is not loaded.")
            return None

        embeddings = []
        for face in faces:
            face = np.expand_dims(face, axis=0)
            feed_dict = {self.input_tensor: face, self.phase_train_tensor: False}
            embedding = self.sess.run(self.output_tensor, feed_dict=feed_dict)
            embedding = np.squeeze(embedding)  # Squeeze to ensure it's 1-D
            embeddings.append(embedding)
        
        print("Embedding complete.")
        return embeddings

    # Train SVM Classifier
    def train_svm_classifier(self):
        if self.known_embeddings is None or self.known_labels is None:
            print("Known embeddings or labels are not available.")
            return
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(self.known_labels)
        
        self.svm_classifier = SVC(kernel='linear', probability=True)
        self.svm_classifier.fit(self.known_embeddings, encoded_labels)
        print("SVM classifier trained.")

    # Recognize Faces using SVM
    def recognize_faces_svm(self, embeddings):
        if self.svm_classifier is None or self.label_encoder is None:
            print("SVM classifier or label encoder is not trained.")
            return None

        recognized_faces = []
        for embedding in embeddings:
            encoded_label = self.svm_classifier.predict([embedding])
            label = self.label_encoder.inverse_transform(encoded_label)[0]
            probability = np.max(self.svm_classifier.predict_proba([embedding]))
            recognized_faces.append((label, probability))
        
        return recognized_faces
    
    # Process Database
    def process_database(self):
        embeddings_file = 'known_embeddings.pkl'
        labels_file = 'known_labels.pkl'

        # Check if the embeddings and labels files exist
        if os.path.exists(embeddings_file) and os.path.exists(labels_file):
            print("Loading known embeddings and labels from files.")
            with open(embeddings_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            with open(labels_file, 'rb') as f:
                self.known_labels = pickle.load(f)
            return
        
        print("Processing database to generate known embeddings and labels.")
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
        self.known_embeddings = self.generate_embeddings(preprocessed_known_faces)

        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.known_embeddings, f)
        with open(labels_file, 'wb') as f:
            pickle.dump(self.known_labels, f)
        
    # Save Result
    def run_recognition(self):
        self.process_database()
        self.train_svm_classifier()
        
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
                embeddings = self.generate_embeddings(preprocessed_faces)

                recognized_faces = self.recognize_faces_svm(embeddings)

                for (startX, startY, endX, endY, _), (label, probability) in zip(faces, recognized_faces):
                    cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 4)
                    cv.putText(image, f'{label} - {round(probability*100, 2)}%', (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
                
                cv.imwrite(f'{self.output}/{basename}_recognition{ext}', image)
            else:
                print("No faces found. Repeating iteration.")
                cv.imwrite(f'{self.output}/{basename}_recognition{ext}', image)


def main():
    FN = Facenet(input='captures/standard/rance', output='captures_rec_svm/standard/rance', database='database')
    FN.run_recognition()
    del FN

    FN = Facenet(input='captures/enhanced/rance', output='captures_rec_svm/enhanced/rance', database='database')
    FN.run_recognition()
    del FN

if __name__ == '__main__':
    main()