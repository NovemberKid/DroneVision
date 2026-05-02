import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Path to wildlife drone footage
video_path = "videos/wildlife_sample4.mp4"

# capture the object and read the video
cap = cv2.VideoCapture(video_path)

# validation if cap can open the video or not
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# dictionary to count detected species
detection_count = {}

while True:
    # Read next frame
    ret, frame = cap.read()

    # Stop if video ends or frame fails
    if not ret:
        print("End of video or failed to read frame.")
        break

    # run YOLO object detection here
    results = model(frame, conf=0.5)

    # this is to draw bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        # get readable class name
        class_name = model.names[class_id]

        # count species detections
        if class_name not in detection_count:
            detection_count[class_name] = 0

        detection_count[class_name] += 1

        label = f"{class_name} {confidence:.2f}"

        # Green color box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Displaying annotated frame
    cv2.imshow("DroneVision Wildlife Feed", frame)

    # ord is basically quit when user presses q
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release video resources
cap.release()
cv2.destroyAllWindows()

# Printing our final wildlife detection summary
print("\nDetection Summary:")
for animal, count in detection_count.items():
    print(f"{animal}: {count}")