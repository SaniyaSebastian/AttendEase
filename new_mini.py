import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
import time
import torch
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import csv
from PIL import Image

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

# Function to detect and recognize faces
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
        if max_similarity > 0.5:
            identity = embeddings_df.iloc[max_similarity_index]['label']
            recognized_faces[identity] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return recognized_faces

# Function to write attendance to CSV
def write_to_csv(recognized_faces):
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Identity', 'Timestamp'])
        for identity, timestamp in recognized_faces.items():
            writer.writerow([identity, timestamp])
    print("Attendance sheet created successfully.")

# # Function to capture photos for registration and generate embeddings
# def capture_photos(video_capture, name, roll_number, parent_folder, embeddings_df, num_photos=10, interval=2):
#     st.write("Please move your head from left to right.")
#     registration_folder = os.path.join(parent_folder, f"{name}_{roll_number}")
#     os.makedirs(registration_folder, exist_ok=True)
#     photos = []
#     embeddings = []
#     for i in range(num_photos):
#         ret, frame = video_capture.read()
#         if ret:
#             # Convert the frame from BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # Resize the frame to a smaller ratio
#             frame_small = cv2.resize(frame_rgb, (320, 240))
#             # Convert the frame to PIL Image
#             pil_image = Image.fromarray(frame_small)
#             # Save the photo with a sequential filename
#             filename = f"photo_{i + 1}.jpg"
#             filepath = os.path.join(registration_folder, filename)
#             pil_image.save(filepath)
#             # Append the photo filepath to the list
#             photos.append(filepath)
#             # Extract embeddings from the captured face
#             embedding = extract_embedding(frame)
#             embeddings.append(embedding)
#             # Display the photo
#             st.image(pil_image, caption=f"Photo {i + 1}/{num_photos}", channels="RGB")
#             # Wait for the interval
#             time.sleep(interval)
    
#     # Convert the embeddings list to a DataFrame
#     new_embeddings_df = pd.DataFrame(embeddings, columns=embeddings_df.columns[1:])
#     # Add 'label' column with the same value for all rows
#     new_embeddings_df['label'] = f"{name}_{roll_number}"
#     # Concatenate the new DataFrame with the existing one
#     embeddings_df = pd.concat([embeddings_df, new_embeddings_df], ignore_index=True)
    
#     # Save the updated embeddings CSV file
#     embeddings_df.to_csv('embedding_aug.csv', index=False)

#     return photos

# def capture_photos(video_capture, name, roll_number, parent_folder, embeddings_df, num_photos=3, interval=2):
#     st.write("Please move your head from left to right.")
#     registration_folder = os.path.join(parent_folder, f"{name}_{roll_number}")
#     os.makedirs(registration_folder, exist_ok=True)
#     photos = []
#     embeddings = []
#     for i in range(num_photos):
#         ret, frame = video_capture.read()
#         if ret:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_small = cv2.resize(frame_rgb, (320, 240))
#             pil_image = Image.fromarray(frame_small)
#             filename = f"photo_{i + 1}.jpg"
#             filepath = os.path.join(registration_folder, filename)
#             pil_image.save(filepath)
#             photos.append(filepath)
#             embedding = extract_embedding(frame, detector, resnet)
#             embeddings.append(embedding)
#             st.image(pil_image, caption=f"Photo {i + 1}/{num_photos}", channels="RGB")
#             time.sleep(interval)
    
#     new_embeddings_df = pd.DataFrame(embeddings, columns=embeddings_df.columns[1:])
#     new_embeddings_df['label'] = f"{name}_{roll_number}"
#     embeddings_df = pd.concat([embeddings_df, new_embeddings_df], ignore_index=True)
    
#     embeddings_df.to_csv('embedding_aug.csv', index=False)

#     return photos




def extract_embedding(image,detector,resnet):
   

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face and extract embedding
    with torch.no_grad():
        # Detect face
        detections = detector.detect_faces(image_rgb)
        if detections:
            # Get the first face detected
            face_box = detections[0]['box']
            x, y, w, h = face_box
            # Extract the face region from the image
            face = image[y:y+h, x:x+w]
            # Resize and preprocess the face
            resized_face = cv2.resize(face, (160, 160))
            normalized_face = resized_face / 255.0
            tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()
            # Generate embedding
            embedding = resnet(tensor_face).detach().numpy().flatten()
            return embedding
        else:
            return None

      

def capture_photos(video_capture, name, roll_number, parent_folder, embeddings_df, num_photos=3, interval=2):
    st.write("Please move your head from left to right.")
    registration_folder = os.path.join(parent_folder, f"{name}_{roll_number}")
    os.makedirs(registration_folder, exist_ok=True)
    photos = []
    embeddings = []
    for i in range(num_photos):
        ret, frame = video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (320, 240))
            pil_image = Image.fromarray(frame_small)
            filename = f"photo_{i + 1}.jpg"
            filepath = os.path.join(registration_folder, filename)
            pil_image.save(filepath)
            photos.append(filepath)
            embedding = extract_embedding(frame,detector,resnet)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                st.warning(f"Failed to extract embedding for photo {i+1}. Please try again.")
            st.image(pil_image, caption=f"Photo {i + 1}/{num_photos}", channels="RGB")
            time.sleep(interval)
        else:
            st.error("Failed to capture photo. Please check your camera connection.")
    
    if embeddings:
        new_embeddings_df = pd.DataFrame(embeddings, columns=embeddings_df.columns[1:])
        new_embeddings_df['label'] = f"{name}_{roll_number}"
        embeddings_df = pd.concat([embeddings_df, new_embeddings_df], ignore_index=True)
        embeddings_df.to_csv('embedding_aug.csv', index=False, mode='a', header=not os.path.exists('embedding_aug.csv'))
    else:
        st.error("No embeddings extracted. Registration failed.")
        return None

    return photos

  

######################
def main():
    st.title("Face Recognition System")

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
                st.write("Registration complete!")

    # Button outside of the form
    if st.button("Register More Students"):
        st.write("Registering more students...")
        st.experimental_rerun()  # Rerun the app to register more students

    # Take Attendance Button
    if st.button("Take Attendance"):
        st.write("Taking attendance...")

        # Load embeddings from the CSV file
        embeddings_df = pd.read_csv("embedding_aug.csv")
        known_embeddings = embeddings_df.iloc[:, 1:].values  # Extract embeddings from DataFrame

        # Debugging: Check if known_embeddings is empty
        if known_embeddings.size == 0:
            st.error("No embeddings found. Make sure to register faces before taking attendance.")
            return

        # Initialize MTCNN detector
        detector = MTCNN()

        # Initialize FaceNet model
        resnet = InceptionResnetV1(pretrained='vggface2').eval()

        cap = cv2.VideoCapture(0)
        start_time = time.time()
        duration = 60

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

    video_capture.release()


if __name__ == "__main__":
    main()
