import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import matplotlib.patches as patches




# Load your dataset
faces_image = np.load('./input_dataset/allam_training_200.npy').astype(np.float32)
faces_image /= 255.0  # Normalize the data

faces_target = np.load('./input_dataset/allam_targets_200.npy')

# Flatten images
faces_data = faces_image.reshape(faces_image.shape[0], -1)  # Flatten each 64x64 image to 4096

# PCA implementation using TensorFlow
class PCA_TF:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, data):
        self.mean = tf.reduce_mean(data, axis=0)
        data_centered = data - self.mean
        s, u, v = tf.linalg.svd(data_centered, compute_uv=True, full_matrices=False)
        self.components = v[:, :self.n_components]

    def transform(self, data):
        data_centered = data - self.mean
        return tf.matmul(data_centered, self.components)

@tf.keras.utils.register_keras_serializable()
class SimpleNN(Model):
    def __init__(self, num_classes=4, **kwargs):
        super(SimpleNN, self).__init__(**kwargs)
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self):
        config = super(SimpleNN, self).get_config()
        config.update({
            'num_classes': self.dense2.units  # Assuming 'num_classes' corresponds to number of units in the last Dense layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Constants
n_components = 50
num_classes = np.unique(faces_target).shape[0]

# Initialize PCA and the neural network
pca_model = PCA_TF(n_components)
nn_model = SimpleNN(num_classes)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.20, random_state=22)

# Fit PCA on training data
pca_model.fit(X_train)

print("pca model result :",pca_model.components)

# Transform data using PCA
X_train_pca = pca_model.transform(X_train)
X_test_pca = pca_model.transform(X_test)

# Compile the neural network model
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
nn_model.fit(X_train_pca, y_train, epochs=10)

# Evaluate the model on test data
test_loss, test_acc = nn_model.evaluate(X_test_pca, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Optionally, add TensorFlow Lite conversion here as needed


# Predict on test data
y_pred = nn_model.predict(X_test_pca)
y_pred_labels = tf.argmax(y_pred, axis=1).numpy()

# Generate and display classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))


# Convert tensor to a NumPy array
components_np = pca_model.components.numpy()
components_np = pca_model.components.numpy()

# Assuming 'pca_model' is your trained PCA instance with 120 components
transformation_matrix = components_np.T  # Transpose of the matrix
mean_vector = pca_model.mean # Shape should be (4096,)

# print("transformation_matrix is  : ", transformation_matrix.shape)
# print("mean vector  : ",mean_vector)

# np.savetxt('transformation_matrix.csv', transformation_matrix, delimiter=',')
# np.savetxt('mean_vector.csv', mean_vector, delimiter=',')

# Convert the TensorFlow Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
tflite_model_path = "./tensorflow_models/nn_model_200.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)






def get_class_name(output_data):
    class_mapping = {
        0: "Adham_Allam",
        1: "Dua_Lipa",
        2: "Henry_Cavill",
        3: "Scarlett_Johansson",
        # Add more mappings if needed
    }
    # Find the index of the class with the highest probability
    max_index = np.argmax(output_data[0])
    print("max_index : ",max_index)
    max_value = output_data[0][max_index]
    print("max value : ",max_value)

    
    # Check if the maximum confidence score is less than 0.8
    if max_value < 0.8:
        return "Unknown"
    else:
        if max_index in class_mapping:
            return class_mapping[max_index]
        else:
            return "Unknown"
# Path to your trained PCA model or configuration to initialize it
# Assuming pca is already loaded or defined elsewhere

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'cam2.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(figsize=(8, 6))  # figsize is in inches

# Convert color space from BGR (OpenCV) to RGB (matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
ax.imshow(image_rgb)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over detected faces and perform inference on each cropped face
for (x, y, w, h) in faces:
    # Crop the face region from the original image
    face_roi = gray[y:y+h, x:x+w]

    # Resize the cropped face region to 64x64 (assuming this was the size used during training)
    resized_face = cv2.resize(face_roi, (64, 64))

    # Convert the resized face to a numpy array and normalize it
    image_array = resized_face.astype(np.float32) / 255.0

    # Flatten the image to create a 1D array
    image_flattened = image_array.reshape(1, -1)  # Shape (1, 4096) if the image is 64x64

    # Apply PCA transformation (ensure the PCA instance is fitted to your training data)
    pca_transformed_image = pca_model.transform(image_flattened)

    # Convert to TensorFlow tensor
    test_input = tf.convert_to_tensor(pca_transformed_image, dtype=tf.float32)

    # Load your TensorFlow Lite model and prepare for inference
    interpreter = tf.lite.Interpreter(model_path="./tensorflow_models/nn_model_200.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to the input of the model
    interpreter.set_tensor(input_details[0]['index'], test_input)

    # Run the model on the input data
    interpreter.invoke()

    # Extract the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Inference Result:", output_data)
    print("Output Data:", output_data)
    print("Shape of Output Data:", output_data.shape)
    class_name = get_class_name(output_data)

    # # Display the cropped face
    # cv2.imshow('Cropped Face', resized_face)
    # cv2.waitKey(0)
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # # Display the predicted class name above the image
    # cv2.putText(image,class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Display the predicted class name above the rectangle
    ax.text(x, y - 10, class_name, color='red', fontsize=12, weight='bold')

    # Display the cropped face
    # cv2.imshow('Cropped Face', resized_face)
    cv2.waitKey(0)

plt.axis('off')  # Hide axes
plt.show()