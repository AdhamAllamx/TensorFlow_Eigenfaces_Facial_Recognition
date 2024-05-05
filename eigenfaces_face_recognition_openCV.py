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
faces_image = np.load('./input_dataset/allam_training_400.npy').astype(np.float32)
faces_image /= 255.0  # Normalize the data

faces_target = np.load('./input_dataset/allam_targets_400.npy')

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
class kNN(Model):
    def __init__(self, num_classes=4, **kwargs):
        super(kNN, self).__init__(**kwargs)
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self):
        config = super(kNN, self).get_config()
        config.update({
            'num_classes': self.dense2.units  # Assuming 'num_classes' corresponds to number of units in the last Dense layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Constants
n_components = 100
num_classes = np.unique(faces_target).shape[0]

# Initialize PCA and the neural network
pca_model = PCA_TF(n_components)
knn_model = kNN(num_classes)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.20, random_state=41)

# Fit PCA on training data
pca_model.fit(X_train)

print("pca model result :",pca_model.components)

# Transform data using PCA
X_train_pca = pca_model.transform(X_train)
X_test_pca = pca_model.transform(X_test)

# Compile the neural network model
knn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
knn_model.fit(X_train_pca, y_train, epochs=10)

# Evaluate the model on test data
test_loss, test_acc = knn_model.evaluate(X_test_pca, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Optionally, add TensorFlow Lite conversion here as needed


# Predict on test data
y_pred = knn_model.predict(X_test_pca)
y_pred_labels = tf.argmax(y_pred, axis=1).numpy()

# Generate and display classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))


# Convert tensor to a NumPy array
components_np = pca_model.components.numpy()
components_np = pca_model.components.numpy()

# converter = tf.lite.TFLiteConverter.from_keras_model(knn_model)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_model = converter.convert()

# open(tflite_model +'.tflite','wb').write(tflite_model)


# Assuming 'pca_model' is your trained PCA instance with 120 components
# transformation_matrix = components_np.T  # Transpose of the matrix
# mean_vector = pca_model.mean # Shape should be (4096,)

# print("transformation_matrix is  : ", transformation_matrix.shape)
# print("mean vector  : ",mean_vector)

# np.savetxt('transformation_matrix_400.csv', transformation_matrix, delimiter=',')
# np.savetxt('mean_vector_400.csv', mean_vector, delimiter=',')

# Convert the TensorFlow Keras model to TensorFlow Lite
# Save the Keras model in the SavedModel format
# Save the Keras model in the SavedModel format
# model_dir = './nn_model_400'
# os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

# model_path = os.path.join(model_dir, 'nn_model_400')
# nn_model.export(model_path)  # Save the model in TensorFlow SavedModel format
# print(f"Model saved successfully at: {model_path}")

# converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
# tflite_model = converter.convert()

# # Save the TensorFlow Lite model to a file
# tflite_path = os.path.join(model_dir, 'nn_model_400.tflite')
# with open(tflite_path, 'wb') as f:
#     f.write(tflite_model)



# # # Load the saved TensorFlow model
# model_path = './nn_model_400/nn_model_400'
# model_dir = './nn_model_400'
# converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

# # Apply optimizations for deployment
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Ensure the converter to use full integer quantization
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# # Assuming X_train_pca is your PCA-transformed training dataset
# # Ensure X_train_pca is loaded and available here, if not loaded already
# # X_train_pca = np.load('./input_dataset/X_train_pca.npy')

# print("Total samples in X_train_pca:", len(X_train_pca))


# # Define a representative dataset generator using actual data

# def representative_dataset_gen():
#     num_samples = min(100, len(X_train_pca))  # Adjust the number of samples based on your dataset size
#     for i in range(num_samples):
#         # Convert the numpy data to a TensorFlow tensor with the correct dtype
#         # Make sure the data is in float32 as expected by TFLite if quantization is applied
#         data = tf.reshape(X_train_pca[i], [1, -1])
#         data = tf.cast(data, tf.float32)  # Ensure the tensor is of type float32
#         yield [data]

# # Set the representative dataset for calibration
# converter.representative_dataset = representative_dataset_gen

# # Ensure all model inputs and outputs are integers
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

# # Convert the model to a TensorFlow Lite model
# tflite_quantized_model = converter.convert()

# # Save the quantized model
# tflite_model_quantized_path = os.path.join(model_dir, 'nn_model_400_quantized.tflite')
# with open(tflite_model_quantized_path, 'wb') as f:
#     f.write(tflite_quantized_model)
#     print(f"Quantized model saved at: {tflite_model_quantized_path}")

import os

# # Path to the TensorFlow Lite model file
# tflite_model_path = './nn_model_400/nn_model_400.tflite'

# # Convert the model to a C header file
# os.system(f"xxd -i {tflite_model_path} > model_data.cc")


# import numpy as np
# import os

# def convert_tflite_to_header(tflite_path, output_header_path):

#     with open(tflite_path, 'rb') as tflite_file:
#         tflite_content = tflite_file.read()


#     hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i+12]]) for i in range(0, len(tflite_content), 12)]


#     hex_array = ',\n  '.join(hex_lines)


#     with open(output_header_path, 'w') as header_file:
        
#         header_file.write('const unsigned char model[] = {\n  ')
#         header_file.write(f'{hex_array}\n')
#         header_file.write('};\n\n')
# tflite_path = './nn_model_400/nn_model_400_quantized.tflite'
# output_header_path = 'nn_model_400_quantized.h'

# convert_tflite_to_header(tflite_path, output_header_path)
# convert_tflite_to_header(tflite_path, output_header_path)



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
image_path = 'sc3.jpg'
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

    result_test = knn_model.predict(pca_transformed_image)

    print("result test ",result_test)


    # Convert to TensorFlow tensor
    # test_input = tf.convert_to_tensor(pca_transformed_image, dtype=tf.float32)

    # # Load your TensorFlow Lite model and prepare for inference
    # interpreter = tf.lite.Interpreter(model_path="./nn_model_400/nn_model_400.tflite")
    # interpreter.allocate_tensors()

    # # Get input and output details
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Set the tensor to the input of the model
    # interpreter.set_tensor(input_details[0]['index'], test_input)

    # # Run the model on the input data
    # interpreter.invoke()

    # # Extract the output tensor
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print("Inference Result:", output_data)
    # print("Output Data:", output_data)
    # print("Shape of Output Data:", output_data.shape)
    # class_name = get_class_name(output_data)

    # # Display the cropped face
    # cv2.imshow('Cropped Face', resized_face)
    # cv2.waitKey(0)
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # # Display the predicted class name above the image
    # cv2.putText(image,class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Display the predicted class name above the rectangle
    ax.text(x, y - 10, result_test[0], color='red', fontsize=12, weight='bold')

    # Display the cropped face
    # cv2.imshow('Cropped Face', resized_face)
    cv2.waitKey(0)

plt.axis('off')  # Hide axes
plt.show()