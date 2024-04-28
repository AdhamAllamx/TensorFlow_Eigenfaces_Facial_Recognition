import numpy as np
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):  # Ensure the files are read in order
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def save_data(images, labels, save_path, prefix):
    # Convert list of images to a numpy array and flatten if necessary
    images = np.array(images)
    if images.ndim == 3:
        images = images.reshape(images.shape[0], -1)  # Flatten the images to 1D per image
    np.save(os.path.join(save_path, f'{prefix}_training.npy'), images)
    np.save(os.path.join(save_path, f'{prefix}_targets.npy'), labels)

# Directory containing the folders for each person
base_path = './input_dataset/augmented_images/'
save_path = './input_dataset/'

# List the folders (one for each person)
persons = ['Adham_Allam', 'Dua_Lipa', 'Henry_Cavil', 'Scarelett_Johansson']
all_images = []

# Generate labels for each person, assuming 100 images per person
labels = np.repeat(np.arange(1, 5), 100)  # Generates [1,1,..., 2,2,..., 3,3,..., 4,4,...] each repeated 100 times

for person in persons:
    folder_path = os.path.join(base_path, person)
    images = load_images_from_folder(folder_path)
    all_images.extend(images)

# Ensure that the number of images matches the number of labels
assert len(all_images) == len(labels), "Number of images and labels must match"

# Save the data
save_data(all_images, labels, save_path, 'allam')

print("Data processed and saved successfully!")
