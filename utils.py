import cv2
from keras.models import load_model
import os
from keras.preprocessing.image import img_to_array
import numpy as np
import face_recognition as fr
import cv2
import face_recognition
import json
import numpy as np
from streamlit_webrtc import VideoTransformerBase
from datetime import datetime
import pandas as pd
import streamlit as st


data_dir = './database'


def append_to_csv(name, id, emotion):
    csv_file = os.path.join(data_dir, 'record.csv')
   
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    new_data = pd.DataFrame({
        'Name': [name],
        'ID': [id],
        'Emotion': [emotion],
        'Time': [current_time]
    })
    
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        existing_data = pd.read_csv(csv_file)
        
        if id in id_details:
            print(f"ID {id} already exists. Data not recorded.")
            return
        else:
            id_details.append(id)
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(csv_file, mode='a', header=True, index=False)
    


def load_encodings():
    with open(data_dir+"/data.json", "r") as f:
        data = json.load(f)

    encodings = data['encodings']
    faces = data['ids']
    names = data['names']

    return encodings, faces, names


class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection logic
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, locations)

        encodings, faces, names = load_encodings()

        for (t, r, b, l), encoding in zip(locations, face_encodings):
            matches = face_recognition.compare_faces(encodings, encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = faces[first_match_index]

            color = (0, 0, 255) if name != "Unknown" else (255, 0, 0)
            cv2.rectangle(img, (l, t), (r, b), color, 2)
            cv2.rectangle(img, (l, b - 35), (r, b), color, cv2.FILLED)
            cv2.putText(img, name, (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return img


def detect_faces(stream_path, id_data):
    global id_details
    id_details = id_data 
    face_classifier, classifier = load_models()
    cap = cv2.VideoCapture(stream_path)
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        encodings, faces, names = load_encodings()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, locations)

        for (t, r, b, l), encoding in zip(locations, face_encodings):
            matches = face_recognition.compare_faces(encodings, encoding)

            name = False
            emotion = 'None'

            distances = face_recognition.face_distance(encodings, encoding)
            match_idx = np.argmin(distances)

            if matches[match_idx]:
                name = faces[match_idx]
                emotion = video_emotion_classification(gray_frame[t:b, l:r], face_classifier, classifier)
                append_to_csv(names[name], str(name), emotion)

            colors = (0, 255, 0) if name else (255, 0, 0)
            cv2.rectangle(frame, (l, t), (r, b), colors, 2)
            cv2.rectangle(frame, (l, b-35), (r, b), colors, cv2.FILLED)
            cv2.putText(frame, str(name), (l+6, b-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            cv2.rectangle(frame, (l, t), (r, t - 35), (0, 140, 255), cv2.FILLED)
            cv2.putText(frame, emotion, (l + 6, t - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)



        frame_resized = cv2.resize(frame, (640, 480))

        # Create a named window
        cv2.namedWindow('Window', cv2.WINDOW_NORMAL)

        # # Resize the window to 640x480
        cv2.resizeWindow('Window', 640, 480)

        # Display the image in the resized window
        cv2.imshow('Window', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True


def encode_image(img_path, id, name):

    with open(data_dir+'/data.json', 'r') as jf:
        data = json.load(jf)

    img = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)[0]
    data['encodings'].append(list(face_encodings))
    data['ids'].append(id)
    data['names'][id] = name

    with open(data_dir+'/data.json', 'w') as jf:
        json.dump(data, jf)


    return True


def load_models():

    face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
    classifier =load_model("./models/model.h5")

    return face_classifier, classifier


def video_emotion_classification(frame, face_classifier, classifier):

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    labels = []
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    # roi_gray = gray[y:y+h,x:x+w]
    roi_gray = frame
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        prediction = classifier.predict(roi)[0]
        print(prediction)
        label=emotion_labels[prediction.argmax()]
        
        return label
    
