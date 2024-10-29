import cv2 as cv
import joblib
import numpy as np
import pandas as pd

print("Deployment script loaded successfully!")

# Load your trained model
print("Loading model...")
model = joblib.load('deployed/hand_gesture_decision_tree.pkl')
print("Model loaded successfully!")

print("Loading preprocessing pipeline...")
pipeline = joblib.load('deployed/preprocessing_pipeline.pkl')
print("Preprocessing pipeline loaded successfully!")

# Define the features based on your trained model
features = ['perimeter', 'solidity', 'circularity', 'eccentricity', 'major_axis_length', 'minor_axis_length']


def getContourFeatures(contour):
    """ Return selected contour features based on contour analysis. """
    # Simple contour features
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    # Convex hull features
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Eccentricity and fitting ellipse parameters
    try:
        ellipse = cv.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2) if major_axis_length > 0 else 0
    except Exception as e:
        major_axis_length = minor_axis_length = eccentricity = 0

    # Create a feature vector with only the selected features
    features = np.array([
        perimeter,
        solidity,
        circularity,
        eccentricity,
        major_axis_length,
        minor_axis_length
    ])
    
    return features

def extract_features_from_frame(frame):
    """ Extract features from the given frame using contour analysis. """
    # Convert the image to the HSV color space
    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask where skin color is within the specified range
    masked_img = cv.inRange(hsv_img, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask (remove small noises)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    masked_img = cv.morphologyEx(masked_img, cv.MORPH_CLOSE, kernel)
    masked_img = cv.erode(masked_img, None, iterations=2)
    masked_img = cv.dilate(masked_img, None, iterations=2)

    # Find contours in the masked image
    contours, _ = cv.findContours(masked_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assuming it's the hand)
        # print("Contours found:", len(contours))
        cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

        hand_contour = max(contours, key=cv.contourArea)
        

        # Extract features from the largest contour
        features = getContourFeatures(hand_contour)  # Use the function you defined earlier

        return features, hand_contour  # Return the extracted features

    return None, None  # Return None if no contour is found


# Start video capture
print("Starting video capture...")
cap = cv.VideoCapture(0)  # 0 for default camera
print("Video capture started successfully!")

label_mapping = {
    0: 'Paper',
    1: 'Rock',
    2: 'Scissors'
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features from the frame
    features_array, hand_contour = extract_features_from_frame(frame)

    # Make predictions if features are available
    if features_array is not None:
        print("Extracted Features:", features_array)

        # Create a DataFrame from the features_array with appropriate column names
        features_df = pd.DataFrame([features_array], columns=features)  # Ensure 'features' has the correct names

        # Transform the features using the pipeline
        transformed_features = pipeline.transform(features_df)  # Now using DataFrame
        print("Transformed Features:", transformed_features)

        # Convert the transformed features back to a DataFrame (if needed)
        features_transformed_df = pd.DataFrame(transformed_features, columns=features)  # Ensure consistency

        # Predict using the model
        prediction = model.predict(features_transformed_df)
        label = prediction[0]  # Get the predicted label

        class_label = label_mapping.get(label, "Unknown")  # Default to "Unknown" if label is not in mapping
        print(f"Prediction: {class_label}")

        # Draw a bounding box around the contour
        x, y, w, h = cv.boundingRect(hand_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Black box

        # Display the label in the box
        cv.putText(frame, f'Prediction: {class_label}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show the frame
    cv.imshow('Live Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(100) & 0xFF == ord('q'):
        print("Video capture stopped.")
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()