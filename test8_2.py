import os
import subprocess
import time
import torch
import cv2
import threading
import queue as Queue

def take_photo(queue, capture_event, detection_event):
    while True:
        detection_event.wait()  # Wait for the detection to complete
        img_path = f"temp_{int(time.time())}.jpg"
        try:
            subprocess.run(["fswebcam", "-r", "1280x720", "--jpeg", "100", "-D", "2", img_path], check=True)
            print("Taking a photo...")
            queue.put(img_path)
            detection_event.clear()  # Clear the detection event for the next cycle
            capture_event.set()  # Signal to start detection
        except subprocess.CalledProcessError as e:
            print(f"Error capturing image: {e}")

def detect_objects(queue, capture_event, detection_event, model, device):
    while True:
        capture_event.wait()  # Wait for a new photo to be taken
        if not queue.empty():
            image_path = queue.get()
            print("Detecting objects in the photo...")
            process_image(image_path, model, device)
            os.remove(image_path)  # Remove the temporary file
            capture_event.clear()  # Clear the capture event for the next cycle
            detection_event.set()  # Signal to capture a new photo

def process_image(image_path, model, device):
    try:
        # Read and convert the image
        frame = cv2.imread(image_path)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        start_time = time.time()
        results = model(img)

        # Process the results
        detections = results.xyxy[0].cpu().numpy()  # xyxy format (x1, y1, x2, y2, confidence, class)

        # Print detection results and save images accordingly
        if any([detection[4] > 0.5 for detection in detections]):
            print("Detected")
            # Label the detected objects in the image
            for detection in detections:
                if detection[4] > 0.5:
                    x1, y1, x2, y2, confidence, class_id = detection
                    label = f"{confidence:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the detected image and annotations
            detected_img_path = f"detections/detected_{int(time.time())}.jpg"
            cv2.imwrite(detected_img_path, frame)
            save_detections(detections, img_path=detected_img_path)
        else:
            print("No detection")
            print("No object detected.")

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

    except Exception as e:
        print(f"Error processing image: {e}")

def save_detections(detections, img_path):
    if not os.path.exists("detections"):
        os.makedirs("detections")

    with open(f"detections/{os.path.basename(img_path)}.txt", "w") as f:
        for detection in detections:
            f.write(f"{detection[0]:.3f}, {detection[1]:.3f}, {detection[2]:.3f}, {detection[3]:.3f}, {detection[4]:.3f}\n")

def main():
    print("Loading YOLOv5 model...")
    # Load the custom-trained YOLOv5 model with the given weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/ns/Downloads/test_pass1_ok.pt')

    print("Setting up device...")
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    q = Queue.Queue()
    capture_event = threading.Event()
    detection_event = threading.Event()

    photo_thread = threading.Thread(target=take_photo, args=(q, capture_event, detection_event))
    detection_thread = threading.Thread(target=detect_objects, args=(q, capture_event, detection_event, model, device))

    photo_thread.start()
    detection_thread.start()

    detection_event.set()  # Signal to start capturing photos

    photo_thread.join()
    detection_thread.join()

if __name__ == "__main__":
    main()
