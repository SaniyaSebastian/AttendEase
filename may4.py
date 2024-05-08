import streamlit as st
import cv2
import requests
from PIL import ImageEnhance
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import numpy as np
import time
import torch
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import csv
import random
from torchvision.transforms import functional as F
from PIL import ImageFilter

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to preprocess image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)
    gamma = 1.5
    corrected_image = np.uint8(cv2.pow(equalized_image / 255.0, gamma) * 255)
    enhanced_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
    return enhanced_image

# # Function to detect and recognize faces
def detect_and_recognize_faces(image, embeddings_df, known_embeddings, detector, resnet):
    detections = detector.detect_faces(image)

    recognized_faces = {}
    for detection in detections:
        x, y, w, h = detection['box']
        face = image[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (160, 160))
        normalized_face = resized_face / 255.0
        tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()
        with torch.no_grad():
            detected_embedding = resnet(tensor_face).detach().numpy().flatten()
        similarities = cosine_similarity([detected_embedding], known_embeddings)
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[0, max_similarity_index]
        if max_similarity > 0.7:
            identity = embeddings_df.iloc[max_similarity_index]['label']
            recognized_faces[identity] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return recognized_faces


# def detect_and_recognize_faces(image, embeddings_df, known_embeddings, detector, resnet):
#     #global recognition_start_time
#     recognition_start_time = None
    
#     detections = detector.detect_faces(image)

#     recognized_faces = {}
#     for detection in detections:
#         x, y, w, h = detection['box']
#         face = image[y:y + h, x:x + w]
#         resized_face = cv2.resize(face, (160, 160))
#         normalized_face = resized_face / 255.0
#         tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()
#         with torch.no_grad():
#             detected_embedding = resnet(tensor_face).detach().numpy().flatten()
#         similarities = cosine_similarity([detected_embedding], known_embeddings)
#         max_similarity_index = np.argmax(similarities)
#         max_similarity = similarities[0, max_similarity_index]
#         if max_similarity > 0.6:
#             identity = embeddings_df.iloc[max_similarity_index]['label']
#             if recognition_start_time is None:
#                 recognition_start_time = time.time()  # Start recognition timer
#             else:
#                 recognition_duration = time.time() - recognition_start_time
#                 if recognition_duration >= 4:  # Face recognized for more than 4 seconds
#                     recognized_faces[identity] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
#                     recognition_start_time = None  # Reset recognition timer
#         else:
#             recognition_start_time = None  # Reset recognition timer if face is not recognized
            
#     return recognized_faces
# Function to write attendance to CSV
def write_to_csv(recognized_faces):
    try:
        with open('attendance.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Identity', 'Timestamp'])
            for identity, timestamp in recognized_faces.items():
                writer.writerow([identity, timestamp])
        print("Attendance sheet created successfully.")
    except PermissionError as e:
        st.error(f"Permission denied: {e}")

# Function to display attendance as a table
def display_attendance_table(attendance):
    if attendance:
        st.subheader("Attendance Table")
        data = []
        for key, value in attendance.items():
            name, roll_number = key.split('_')
            data.append({'Name': name, 'Roll Number': roll_number, 'Timestamp': value})
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.warning("No attendance recorded yet.")

# Function to extract embedding from an image
def extract_embedding(image, detector, resnet):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    if detections:
        face_box = detections[0]['box']
        x, y, w, h = face_box
        face = image[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (160, 160))
        normalized_face = resized_face / 255.0
        tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()
        embedding = resnet(tensor_face).detach().numpy().flatten()
        return embedding
    else:
        return None

def capture_photos(video_capture, name, roll_number, parent_folder, embeddings_df, num_photos_per_person=15, interval=3):
    st.write("Please move your head from left to right.")
    registration_folder = os.path.join(parent_folder, f"{name}_{roll_number}")
    os.makedirs(registration_folder, exist_ok=True)
    photos = []
    embeddings = []
    total_photos = num_photos_per_person  # Total photos to capture for each person

    # Loop to capture photos
    for i in range(total_photos):
        ret, frame = video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (320, 240))
            pil_image = Image.fromarray(frame_small)

            # Save default photo
            filename = f"photo_{i + 1}.jpg"
            filepath = os.path.join(registration_folder, filename)
            pil_image.save(filepath)
            photos.append(filepath)

            # Generate embedding for default photo
            embedding = extract_embedding(np.array(pil_image), detector, resnet)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                st.warning(f"Failed to extract embedding for photo {i+1}. Please try again.")
            
            # Display default photo
            st.image(pil_image, caption=f"Photo {i + 1}/{total_photos}", channels="RGB", width=150)
            time.sleep(interval)

            if len(photos) < total_photos:
                # Apply augmentations and save augmented photo if total photos are not reached yet
                augmented_image = apply_augmentations(pil_image)
                augmented_filename = f"photo_{i + 1}_augmented.jpg"
                augmented_filepath = os.path.join(registration_folder, augmented_filename)
                augmented_image.save(augmented_filepath)

                # Generate embedding for augmented photo
                augmented_embedding = extract_embedding(np.array(augmented_image), detector, resnet)
                if augmented_embedding is not None:
                    embeddings.append(augmented_embedding)
                else:
                    st.warning(f"Failed to extract embedding for augmented photo {augmented_filename}. Please try again.")

        else:
            st.error("Failed to capture photo. Please check your camera connection.")
    
    # Save embeddings to DataFrame
    if embeddings:
        new_embeddings_df = pd.DataFrame(embeddings, columns=embeddings_df.columns[1:])
        new_embeddings_df['label'] = f"{name}_{roll_number}"
        embeddings_df = pd.concat([embeddings_df, new_embeddings_df], ignore_index=True)
        embeddings_df.to_csv('embedd_iphone.csv', index=False, mode='a', header=not os.path.exists('embedd_iphone.csv'))
    else:
        st.error("No embeddings extracted. Registration failed.")
        return None

    return photos


def apply_augmentations(image):
    # Random rotation (-10 to 10 degrees)
    angle = random.uniform(-10, 10)
    image = image.rotate(angle)

    # Random scaling (95% to 105%)
    scale_factor = random.uniform(0.95, 1.05)
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    image = image.resize((new_width, new_height))

    # Random Gaussian blur (kernel size 1x1 to 3x3)
    kernel_size = random.choice([1, 3])
    image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    # Random brightness adjustment (90% to 110%)
    brightness_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Random contrast adjustment (90% to 110%)
    contrast_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Random color saturation adjustment (90% to 110%)
    saturation_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)

    # Random sharpness adjustment (90% to 110%)
    sharpness_factor = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    return image

def main():
    st.title("Face Recognition System")
    recognition_start_time = None 
    parent_folder = "registered_photos"
    os.makedirs(parent_folder, exist_ok=True)

    embeddings_df = pd.DataFrame(columns=['label'] + [f'emb_{i}' for i in range(512)])  # Initialize embeddings DataFrame
    video_capture = cv2.VideoCapture(0)

    # Register Button
    with st.form("registration_form"):
        st.write("Welcome to Registration")
        name = st.text_input("Enter Name", key="name_input")
        roll_number = st.text_input("Enter Roll Number", key="roll_number_input")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if name and roll_number:
                photos = capture_photos(video_capture, name, roll_number, parent_folder, embeddings_df)
                if photos is not None:
                    st.write("Registration complete!")

    if st.button("Register More Students"):
        st.write("Registering more students...")
        st.experimental_rerun()

    if st.button("Take Attendance"):
        st.write("Taking attendance...")

        embeddings_df = pd.read_csv("embedd_iphone.csv")
        #embeddings_df = pd.read_csv("embeddings_abs.csv")
        known_embeddings = embeddings_df.iloc[:, 1:].values

        if known_embeddings.size == 0:
            st.error("No embeddings found. Make sure to register faces before taking attendance.")
            return

        detector = MTCNN()
        resnet = InceptionResnetV1(pretrained='vggface2').eval()

        #cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('http://192.168.75.111:8080/video')  #VINSU IN sanIya's wifi
        cap = cv2.VideoCapture('http://192.168.75.231:8080/video')  #shreya IN sanIya's wifi
        start_time = time.time()
        duration = 30


        recognized_faces = {}
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break

                preprocessed_frame = preprocess_image(frame)
                recognized_faces.update(detect_and_recognize_faces(preprocessed_frame, embeddings_df,
                                                                   known_embeddings, detector, resnet))
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        write_to_csv(recognized_faces)
        display_attendance_table(recognized_faces)
        

        registered_students = set(embeddings_df['label'])
        recognized_students = set(recognized_faces.keys())
        absentees = registered_students - recognized_students
        if absentees:
            st.subheader("Absentees")
            st.write(", ".join(absentees))
        else:
            st.info("No absentees.")

if __name__ == "__main__":
    main()
