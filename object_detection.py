import cv2
import numpy as np
import time
import os
import yt_dlp as ytdlp

# Function to download video from YouTube using yt-dlp
def download_video(url, output_path="input_video.mp4"):
    print("Downloading video...")
    
    # Set download options to get the best video quality (no audio)
    ydl_opts = {
        'format': 'bestvideo',  # Download only the video (no audio)
        'outtmpl': output_path  # Save as input_video.mp4
    }
    
    # Download the video
    with ytdlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"Video downloaded as {output_path}")
    return output_path

# Load YOLO model
def load_yolo():
    # Replace with your own path to the YOLO files
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Perform object detection on a single frame
def detect_objects(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    detection_time = time.time() - start
    return outs, detection_time

# Process the video and perform object detection
def process_video(video_path):
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_detection_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        outs, detection_time = detect_objects(frame, net, output_layers)
        total_detection_time += detection_time
        frame_count += 1

        # Draw bounding boxes
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    label = str(classes[class_id])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        # Display the frame
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    print(f"Average detection time: {total_detection_time / frame_count:.4f} seconds per frame.")

if __name__ == "__main__":
    # YouTube video URL
    youtube_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB"
    
    # Download the video
    video_path = "input_video.mp4"
    if not os.path.exists(video_path):
        video_path = download_video(youtube_url, video_path)
    
    # Process the video for object detection
    process_video(video_path)
