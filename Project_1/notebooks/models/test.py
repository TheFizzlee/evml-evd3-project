import cv2

def list_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    cameras = list_cameras()
    if cameras:
        print("Available camera indices:", cameras)
    else:
        print("No cameras found.")
