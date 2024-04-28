


# Save the trained models
import pickle
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv


n_row , n_col = 64 , 64
models_file = "trained_data_model.pkl"
# Load the trained models
with open(models_file, "rb") as f:
    pca, iso_forest, clf = pickle.load(f)

    # Dictionary for person names
    person_names = {
        1: "Alvaro_Uribe",
        2: "Atal_Bihari_Vajpayee",
        3: "George_Robertson",
        4: "George_W_Bush",
        5: "Junichiro_Koizumi",
        6: "Adham_Allam"
    }

        

    # Prepare an external image for prediction
    test_image_path = "messi.png"
    test_image = cv2.imread(test_image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if test_image is None:
        print("Test image not found!")
    else:
        # Convert to grayscale for face detection
        gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=13, minSize=(30, 30))
        if len(faces) == 0:
            print("No face detected in the image.")
        else:
            x, y, w, h = faces[0]  # Assume the first detected face is the target face
            is_outlier_list = []
            for (x,y,w,h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(test_image,(x,y),(x+w, y+h),(255, 0,0),2)
                # Crop the face using detected coordinates
                face = gray_test_image[y:y+h, x:x+w]
                # Display the original image with detected face marked
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
                plt.title("Detected Face")
                plt.axis('off')
                # Display the cropped face
                plt.subplot(1, 2, 2)
                plt.imshow(face, cmap='gray')
                plt.title("Cropped Face")
                plt.axis('off')
                plt.show()

            face = gray_test_image[y:y+h,x:x+w]  # Crop the face
            # Resize the face to the desired dimensions
            resized_face = resize(face, (n_row, n_col), anti_aliasing=True).reshape(1,-1)
            # PCA transformation
            test_image_pca = pca.transform(resized_face)
            # Anomaly detection
            is_outlier = iso_forest.predict(test_image_pca)
            print("outlier : ", is_outlier)
            is_outlier_list.append(is_outlier)  # Store the outlier prediction for this image
            # To save the predictions and their probabilities
            predictions_file = "predictions.csv"
            fieldnames = ["Image", "Predicted Person", "Probability"]
            with open(predictions_file, "a", newline='') as f:  # Use "a" mode for appending to the file
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                 # If the file is empty, write the header
                if f.tell() == 0:
                    writer.writeheader()
                for i, is_outlier in enumerate(is_outlier_list):
                    if is_outlier == -1:
                        predicted_person = "Unknown"
                        predicted_probability = 0.0  # Assuming unknown people have zero probability
                    else:
                        probabilities = clf.predict_proba(test_image_pca)[0]
                        predicted_probability = np.max(probabilities)
                        if predicted_probability < 0.40:
                                predicted_person = "Unknown"
                                predicted_probability = 0.0  # If probability is too low, consider it unknown
                        else:
                                predicted_person_id = np.argmax(probabilities) + 1
                                predicted_person = person_names.get(predicted_person_id, "Unknown")
            
                    # Write the prediction and probability to the file
                    writer.writerow({"Image": test_image_path , "Predicted Person": predicted_person, "Probability": f"{predicted_probability:.2f}"})

    # Display results
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(test_image, f"{predicted_person} ({predicted_probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Person: {predicted_person}")
    plt.axis('off')
    plt.show()