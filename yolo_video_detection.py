import cv2
import numpy as np

# Function to load YOLOv3 model
def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getLayers()]
    
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

# Function to perform object detection on the video
def detect_objects_in_video(video_path, net, output_layers, classes):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        height, width, channels = frame.shape
        boxes, confidences, class_ids = [], [], []
        
        # Process each detection
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw bounding boxes and labels
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with detected objects
        cv2.imshow('Object Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # File paths for YOLOv3
    cfg_path = 'yolov3.cfg'         # Path to YOLOv3 config
    weights_path = 'yolov3.weights'  # Path to YOLOv3 weights
    names_path = 'coco.names'       # Path to class names
    
    # Load YOLOv3 model
    net, output_layers, classes = load_yolo_model(cfg_path, weights_path, names_path)
    
    # Provide the path to your downloaded YouTube video
    video_path = 'downloaded_video.mp4'  # Path to your downloaded video
    
    # Perform object detection
    detect_objects_in_video(video_path, net, output_layers, classes)

if __name__ == "__main__":
    main()
