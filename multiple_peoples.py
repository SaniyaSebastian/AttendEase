#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install facenet-pytorch


# In[2]:


import os
import csv
import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import torch



# In[3]:


# Initialize MTCNN for face detection
mtcnn = MTCNN()


# In[4]:


# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()



# In[5]:


def extract_face(image, box):
    x, y, w, h = [int(coord) for coord in box]
    face = image[y:y+h, x:x+w]
    return face


# In[6]:


import cv2
import pandas as pd
from facenet_pytorch import MTCNN

def preprocess_images(csv_file, batch_size=32):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Lists to store embeddings and identities
    embeddings = []
    identities = []

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Iterate through rows and preprocess images in batches
    batch_images = []
    batch_identities = []
    for index, row in df.iterrows():
        image_path = row['image_path']
        identity = row['label']

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image:", image_path)
            continue

        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to a smaller size
        resized_image = cv2.resize(rgb_image, (160, 160))

        # Append the resized image and identity to the batch
        batch_images.append(resized_image)
        batch_identities.append(identity)

        # If the batch is full or we've reached the end of the dataset, process the batch
        if len(batch_images) == batch_size or index == len(df) - 1:
            # Detect faces using MTCNN
            batch_boxes, _ = mtcnn.detect(batch_images)

            # Iterate through detected faces in the batch
            for i, boxes in enumerate(batch_boxes):
                if boxes is not None:
                    for box in boxes:
                        # Extract face coordinates
                        x, y, w, h = [int(coord) for coord in box]

                        # Extract face and resize
                        face = batch_images[i][y:y+h, x:x+w]
                        resized_face = cv2.resize(face, (160, 160))

                        # Convert to tensor and normalize
                        normalized_face = resized_face / 255.0
                        tensor_face = torch.from_numpy(normalized_face.transpose((2, 0, 1))).unsqueeze(0).float()

                        # Generate embedding using FaceNet
                        with torch.no_grad():
                            embedding = resnet(tensor_face).detach().numpy()

                        # Store embedding and identity
                        embeddings.append(embedding.flatten())
                        identities.append(batch_identities[i])

                        # Show detected face (optional)
                        # cv2.imshow('Detected Face', resized_face)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                else:
                    print("No faces detected in:", image_path)

            # Clear the batch for the next iteration
            batch_images = []
            batch_identities = []

    # Convert embeddings and identities to DataFrame
    embeddings_df = pd.DataFrame(embeddings)
    identities_df = pd.DataFrame(identities, columns=['label'])

    # Concatenate embeddings and identities
    result_df = pd.concat([identities_df, embeddings_df], axis=1)

    if not result_df.empty:
        try:
            # Write DataFrame to CSV file
            csv_output_file = "embedding_modified.csv"  # Output CSV file path
            result_df.to_csv(csv_output_file, index=False)
            print("Embeddings saved to:", csv_output_file)
        except Exception as e:
            print("Error occurred while saving embeddings:", e)
    else:
        print("No embeddings to save.")

    return result_df


# In[7]:


# Your code to create the training_data.csv file goes here
# For example:
data_dir = "C:\\Users\\Admin\\Desktop\\MINIPROJECT\\training_data"
data_rows = []

for student_dir in os.listdir(data_dir):
    student_path = os.path.join(data_dir, student_dir)
    # Check if student_path is a directory
    if os.path.isdir(student_path):
        # Iterate through image files in the student's directory
        for img_file in os.listdir(student_path):
            # Append data row with image path and label
            img_path = os.path.join(student_path, img_file)
            data_rows.append([img_path, student_dir])
    else:
        print(f"Warning: {student_path} is not a valid directory.")

# Write the data rows to the CSV file
csv_file = "training_data1.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['image_path', 'label'])
    # Write data rows
    writer.writerows(data_rows)
print("CSV file - training data created successfully.")


# In[8]:


# Call the function with the path to your CSV file containing image paths and identities
preprocess_images("training_data1.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




