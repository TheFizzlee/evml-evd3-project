import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class ResizeWithAspectRatio:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        wpercent = (self.size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((self.size, hsize), Image.Resampling.LANCZOS)

class AutoExposure:
    def __call__(self, img):
        # Convert the image to a numpy array
        img_array = np.array(img)

        # Split into R, G, B channels
        r, g, b = cv.split(img_array)

        # Apply CLAHE to each channel
        clahe = cv.createCLAHE(clipLimit=25.0, tileGridSize=(8, 8))
        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)

        # Merge the channels back together
        img_array = cv.merge([r, g, b])

        # Convert back to PIL Image
        return Image.fromarray(img_array)

print("Deployment script loaded successfully!")

# Load the trained CNN model
print("Loading model...")
model = torch.load('model/model.pth', map_location=torch.device('cpu'))  # Load the model on CPU
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

print("Loading preprocessing pipeline...")
pipeline = torch.load('model/preprocessing_pipeline.pth', map_location=torch.device('cpu'))  # Load the preprocessing pipeline
print("Preprocessing pipeline loaded successfully!")

# Label mapping
label_mapping = {
    0: 'Paper',
    1: 'Rock',
    2: 'Scissors'
}

# Start video capture
print("Starting video capture...")
cap = cv.VideoCapture(0)  # 0 for default camera
print("Video capture started successfully!")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame using the pipeline
    try:
        # Convert the frame to RGB as PIL expects RGB format
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert to PIL image
        pil_img = Image.fromarray(frame_rgb)

        # Apply the preprocessing pipeline
        processed_img = pipeline(pil_img)

        # Add batch dimension (1, C, H, W)
        input_tensor = processed_img.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        class_label = label_mapping.get(label, "Unknown")  # Default to "Unknown" if label is not in mapping
        print(f"Prediction: {class_label}")

        # Display the prediction on the frame
        cv.putText(frame, f'Prediction: {class_label}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error processing frame: {e}")

    # Show the frame
    # Display the frame using matplotlib
    cv.imshow('Frame', frame)

    # Convert PyTorch tensor to NumPy array
    processed_img_np = processed_img.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC

    # Convert from RGB to BGR if needed (PyTorch tensors often follow RGB convention)
    processed_img_bgr = cv.cvtColor(processed_img_np, cv.COLOR_RGB2BGR)

    # Display using OpenCV
    cv.imshow('Preprocessed Frame', processed_img_bgr)


    # Break the loop on 'q' key press
    if cv.waitKey(100) & 0xFF == ord('q'):
        print("Video capture stopped.")
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()
