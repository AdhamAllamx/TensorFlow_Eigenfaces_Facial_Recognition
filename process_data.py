import numpy as np
import cv2
from skimage.transform import resize
def preprocess_images(images, n_row=64, n_col=64, save_path='./input_dataset/gray_scale_images/'):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    processed_images = np.zeros((len(images), n_row, n_col))  # Corrected to create a 3D array
    for i, img_path in enumerate(images):
        formatted_path = img_path.replace('/', '\\')  # Correct path format for Windows
        img = cv2.imread(formatted_path)
        if img is None:
            raise FileNotFoundError(f"Image at {formatted_path} could not be loaded. Check the file path.")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(faces) == 0:
            raise ValueError(f"No face detected in the image at {formatted_path}.")
        
        # Assume the first detected face is the target face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]  # Crop the face
        
        # Resize the face to the desired dimensions
        resized_face = resize(face, (n_row, n_col), anti_aliasing=True)
        
        # Save the processed image
        save_filename = f'{save_path}processed_{i}.png'
        cv2.imwrite(save_filename, (resized_face * 255).astype(np.uint8))  # Convert back to 0-255 range and cast to uint8

        processed_images[i] = resized_face  # Assign the resized image directly
    
    return processed_images

# Note: rest of the code remains the same.



def append_and_save_data(new_images, new_targets, dataset_path):
    existing_images = np.load(dataset_path + 'olivetti_faces.npy')
    existing_targets = np.load(dataset_path + 'olivetti_faces_target.npy')
    # new_dataset_images = np.load(dataset_path + 'new_faces_training.npy')
    # new_dataset_targets = np.load(dataset_path + 'new_faces_targets.npy')


    # Ensure both datasets are in the same shape format
    if len(existing_images.shape) == 2 and len(new_images.shape) == 3:
        new_images = new_images.reshape(new_images.shape[0], -1)  # Flatten new images if necessary
    elif len(existing_images.shape) == 3 and len(new_images.shape) == 2:
        existing_images = existing_images.reshape(existing_images.shape[0], -1)  # Flatten existing images if necessary

    # all_images = np.concatenate(( new_images), axis=0)
    # all_targets = np.concatenate(( new_targets), axis=0)
    all_images = new_images
    all_targets = new_targets

    np.save(dataset_path + 'new_faces_training.npy', all_images)
    np.save(dataset_path + 'new_faces_targets.npy', all_targets)
# List of new image paths
new_images = [
    # './input_dataset//henry_cavil_1.png',
    # './input_dataset/henry_cavil/henry_cavil_2.jpg',
    # './input_dataset/henry_cavil/henry_cavil_3.jpeg',
    # './input_dataset/henry_cavil/henry_cavil_4.jpeg',
    # './input_dataset/henry_cavil/henry_cavil_5.jpeg',

    # './input_dataset/Adham_Allam/adham1.png',
    # './input_dataset/Adham_Allam/allam6.png',
    #     './input_dataset/Adham_Allam/allam7.png',
            # './input_dataset/Adham_Allam/allam8.png',
            './input_dataset/Adham_Allam/allam8.png',
            # './input_dataset/Adham_Allam/Adham_Allam_0004.png',





    # './input_dataset/scarelett_johansson/scarelett_johansson_1.jpeg',
    # './input_dataset/scarelett_johansson/scarelett_johansson_2.jpeg',
    # './input_dataset/scarelett_johansson/scarelett_johansson_3.jpg',
    # './input_dataset/scarelett_johansson/scarelett_johansson_4.jpg',
    # './input_dataset/scarelett_johansson/scarelett_johansson_5.jpg'

]

save_path = './input_dataset/gray_scale_images/'

new_labels = np.array([9,9,9,9,9,9,9,9,9,9,
                       10,10,10,10,10,10,10,10,10,10
                    ])  # Assuming label 40 for all new images

# Process the new images
preprocessed_new_images = preprocess_images(new_images)

# Define the path for the dataset
dataset_path = './input_dataset/'

# Append and save the new data
append_and_save_data(preprocessed_new_images, new_labels, dataset_path)

print("New Data processed Successfully !!")
