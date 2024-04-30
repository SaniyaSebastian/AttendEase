#!/usr/bin/env python
# coding: utf-8

# 17 APRIL 2024
# includes augmentation + preprocessing + atttendance csv file Input is "embedding_aug.csv"
# 

# In[14]:



# In[1]:


import cv2
import numpy as np
import time
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch  # Add this import statement for torch
import csv  # Add this import statement for CSV writing
import os  # Add this import statement to check file existence


# In[2]:


# Load embeddings from the CSV file
embeddings_df = pd.read_csv("embedding_aug.csv")
known_embeddings = embeddings_df.iloc[:, 1:].values  # Extract embeddings from DataFrame


# In[3]:


# Initialize MTCNN detector
detector = MTCNN()


# In[4]:


# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# In[5]:


#ATTENDANCE SHEET
# Initialize dictionary to track recognized identities
recognized_faces = {}

# Define a function to update recognized identities
def update_recognized_faces(identity):
    # Get current time
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # Update recognized_faces dictionary with identity and timestamp
    recognized_faces[identity] = current_time

# At the end of the video capture loop or when appropriate, write recognized faces to CSV
def write_to_csv():
    with open('attendance_after_aug.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Identity', 'Timestamp'])
        for identity, timestamp in recognized_faces.items():
            writer.writerow([identity, timestamp])
    print("Attendance sheet created successfully.")

def write_to_excel():
    # Convert recognized_faces dictionary to DataFrame
    df = pd.DataFrame(recognized_faces.items(), columns=['Identity', 'Timestamp'])
    # Write DataFrame to Excel file
    excel_output_file = "attendance_aug.xlsx"  # Output Excel file path
    df.to_excel(excel_output_file, index=False)
    print("Attendance sheet created successfully.")


# In[6]:


def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply histogram equalization for better contrast
    equalized_image = cv2.equalizeHist(blurred_image)
    
    # Apply gamma correction for adjusting brightness
    gamma = 1.5
    corrected_image = np.uint8(cv2.pow(equalized_image / 255.0, gamma) * 255)
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
    
    return enhanced_image


# In[7]:





# DELAY OF 4 SECS

# In[8]:


import time

def detect_and_recognize_faces(image):
    global recognition_start_time  # Make recognition_start_time global
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    detections = detector.detect_faces(image_rgb)
    
    # Initialize recognition start time if not already initialized
    if 'recognition_start_time' not in globals():
        recognition_start_time = None
    
    # Draw bounding boxes around detected faces
    for detection in detections:
        x, y, w, h = detection['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 100, 203), 2) # Draw bounding box
        
        # Extract face and resize
        face = image_rgb[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (160, 160))
        
        # Convert to tensor and normalize
        normalized_face = resized_face / 255.0
        tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()
        
        # Generate embedding using FaceNet
        with torch.no_grad():
            detected_embedding = resnet(tensor_face).detach().numpy().flatten()
        
        # Compare detected embedding with known embeddings
        similarities = cosine_similarity([detected_embedding], known_embeddings)
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[0, max_similarity_index]
        
        # Determine identity based on similarity threshold
        if max_similarity > 0.5:  # Adjust threshold as needed
            identity = embeddings_df.iloc[max_similarity_index]['label']
            text = f"Id: {identity}"
            
            # Recognition occurred, start or update the recognition timer
            if recognition_start_time is None:
                recognition_start_time = time.time()
            else:
                recognition_duration = time.time() - recognition_start_time
                if recognition_duration >= 4:  # Adjust time threshold as needed
                    update_recognized_faces(identity)  # Update recognized identities
                    attendance_marked = True  # Set a flag indicating attendance is marked
                    recognition_start_time = None  # Reset recognition start time
        
        else:
            text = "Id: Unknown"
            
            # Reset recognition start time if recognition fails
            recognition_start_time = None
        
        # Draw text on the image
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return image


# In[9]:


# Capture video from webcam
cap = cv2.VideoCapture(0) 


# In[10]:


# Define start time and duration
start_time = time.time()  # Get the current time
duration = 60 # Capture video for 0 seconds


# In[11]:


try:
    while time.time() - start_time < duration:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and recognize faces in the frame
        frame_with_faces = detect_and_recognize_faces(frame)
        
        # Display the frame with bounding boxes and identities
        cv2.imshow('Video', frame_with_faces)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# In[12]:


write_to_csv()
write_to_excel()






