import streamlit as st
import cv2
import time
import os
from PIL import Image

def capture_photos(video_capture, name, roll_number, parent_folder, num_photos=10, interval=2):
    st.write("Please move your head from left to right.")
    registration_folder = os.path.join(parent_folder, f"{name}_{roll_number}")
    os.makedirs(registration_folder, exist_ok=True)
    photos = []
    for i in range(num_photos):
        ret, frame = video_capture.read()
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to a smaller ratio
            frame_small = cv2.resize(frame_rgb, (320, 240))
            # Convert the frame to PIL Image
            pil_image = Image.fromarray(frame_small)
            # Save the photo with a sequential filename
            filename = f"photo_{i+1}.jpg"
            filepath = os.path.join(registration_folder, filename)
            pil_image.save(filepath)
            # Append the photo filepath to the list
            photos.append(filepath)
            # Display the photo
            st.image(pil_image, caption=f"Photo {i+1}/{num_photos}", channels="RGB")
            # Wait for the interval
            time.sleep(interval)
    return photos

def view_attendance():
    name = st.text_input("Enter Name")
    roll_number = st.text_input("Enter Roll Number")
    if st.button("View Attendance"):
        # Logic to view attendance
        st.write(f"Attendance for {name} with Roll Number {roll_number} will be displayed here.")

def main():
    st.title("AttendEase")

    # Create a parent folder to store all registrations
    parent_folder = "captured_photos"
    os.makedirs(parent_folder, exist_ok=True)

    # Create a video capture object
    video_capture = cv2.VideoCapture(0)

    # Add a button to start capturing photos
    if st.button("Start Capturing"):
        # Capture the photos
        name = st.text_input("Enter Name")
        roll_number = st.text_input("Enter Roll Number")
        if name and roll_number:
            photos = capture_photos(video_capture, name, roll_number, parent_folder)
            st.write("Registration complete!")
        else:
            st.error("Please enter Name and Roll Number.")

    # Add a button to view attendance
    view_attendance()

    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    main()
