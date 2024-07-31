import os
import torch
import cv2
import numpy as np
import time
import threading
import tempfile
import queue as Queue

def save_detections(detections, img_path):
    if not os.path.exists("detections"):
        os.makedirs("detections")

    with open(f"detections/{os.path.basename(img_path)}.txt", "w") as f:
        for detection in detections:
            f.write(f"{detection[0]:.3f}, {detection[1]:.3f}, {detection[2]:.3f}, {detection[3]:.3f}, {detection[4]:.3f}\n")

def process_image(image_path, model, device):
    # Read and convert the image
    frame = cv2.imread(image_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Process the results
    detections = results.xyxy[0].cpu().numpy()  # xyxy format (x1, y1, x2, y2, confidence, class)

    # Print detection results and save images accordingly
    timestamp = time.time()
    if any([detection[4] > 0.3 for detection in detections]):  # If an object is detected
        print("Detected")
        # Label the detected objects in the image
        for detection in detections:
            if detection[4] > 0.6:
                x1, y1, x2, y2, confidence, class_id = detection
                label = f"{confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

        # Save the detected image and annotations
        detected_img_path = f"detections/detected_{int(timestamp)}.jpg"
        cv2.imwrite(detected_img_path, frame)
        save_detections(detections, img_path=detected_img_path)
    else:
        print("No detection")

    # Calculate and display FPS
    fps = 1 / (time.time() - timestamp)
    print(f"FPS: {fps:.2f}")

def take_photo(queue, event):
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            img_path = temp_file.name
            cv2.imwrite(img_path, frame)  # Save the captured image to a temporary file

            print("Taking a photo...")
            queue.put(img_path)
            event.set()  # Signal that a new photo is ready
            event.clear()  # Clear the event for the next cycle

    cap.release()

def detect_objects(queue, event, model, device):
    while True:
        event.wait()  # Wait for a new photo to be taken
        if not queue.empty():
            image_path = queue.get()
            print("Detecting objects in the photo...")
            process_image(image_path, model, device)
            os.remove(image_path)  # Remove the temporary file

def main():
    # Load the custom-trained YOLOv5 model with the given weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/ns/Videos/best_4000_last.pt')

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    q = Queue.Queue()
    event = threading.Event()

    photo_thread = threading.Thread(target=take_photo, args=(q, event))
    detection_thread = threading.Thread(target=detect_objects, args=(q, event, model, device))

    photo_thread.start()
    detection_thread.start()

    photo_thread.join()
    detection_thread.join()

if __name__ == "__main__":
    main()
