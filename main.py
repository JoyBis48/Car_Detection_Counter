import cv2
from ultralytics import YOLO
import argparse
from utilities import get_car_color, get_person_gender
import os

# Constants
CONFIDENCE_THRESHOLD = 0.5
CAR_CLASS_ID = 2 # In the pre-trained coco dataset, 2 is the class index for the class 'car'
PERSON_CLASS_ID = 0 # 0 is the class index for the class 'people'
LINE_POSITION = 300
RECTANGLE_COLOR = (0, 255, 0)
RECTANGLE_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2
COUNT_TEXT_POSITION = (10, 50)
PERSON_COUNT_TEXT_POSITION = (10, 70)  
COUNT_TEXT_SCALE = 1
COUNT_TEXT_COLOR = (0, 0, 255)
COUNT_TEXT_THICKNESS = 2
ESC_KEY = 27
DEFAULT_VIDEO_PATH = 'video.mp4'
OUTPUT_VIDEO_PATH = 'run.mp4'

# setting up the arguments for video path and view mode customization via CLI
def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('--video', type=str, help='Path to video file', default=DEFAULT_VIDEO_PATH)
    parser.add_argument('--mode', type=str, choices=['view', 'save'], help='Mode to run the script in', default='view')
    return parser.parse_args()

def initialize_components(video_path):
    if not os.path.exists(video_path):
        print('The video path is invalid')
        return None, None
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    print(f"Loading video from '{video_path}' ")
    return model, cap

# Function to process each frame of the video
def process_frame(frame, model, line_position, car_count, male_count, female_count, person_count, counted_ids, gender_model):
    frame = cv2.resize(frame, (600, 400))
    height, width, _ = frame.shape
    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)

        for i in range(len(boxes)):
            label = ''
            if confidences[i] > CONFIDENCE_THRESHOLD:
                x, y, x2, y2 = map(int, boxes[i])
                tid = track_ids[i]

                if class_ids[i] == CAR_CLASS_ID:
                    roi = frame[y:y2, x:x2]
                    car_color = get_car_color(roi)
                    label = f"{car_color} Car"
                    if y2 > line_position and y < line_position and tid not in counted_ids:
                        car_count += 1
                        counted_ids.add(tid)

                elif class_ids[i] == PERSON_CLASS_ID:
                    roi = frame[y:y2, x:x2]
                    person_gender = get_person_gender(roi, gender_model)
                    label = f"{person_gender}"
                    if y2 > line_position and y < line_position and tid not in counted_ids:
                        if label == "Male":
                            male_count += 1
                        elif label == "Female":
                            female_count += 1
                        else:
                            person_count += 1
                        counted_ids.add(tid)

                cv2.rectangle(frame, (x, y), (x2, y2), RECTANGLE_COLOR, RECTANGLE_THICKNESS)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    cv2.line(frame, (0, line_position), (width, line_position), LINE_COLOR, LINE_THICKNESS)
    cv2.putText(frame, f"Cars: {car_count}", COUNT_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, COUNT_TEXT_SCALE, COUNT_TEXT_COLOR, COUNT_TEXT_THICKNESS)
    person_count_position = (PERSON_COUNT_TEXT_POSITION[0], PERSON_COUNT_TEXT_POSITION[1] + 30)
    cv2.putText(frame, f"People: {person_count}", person_count_position, cv2.FONT_HERSHEY_SIMPLEX, COUNT_TEXT_SCALE, COUNT_TEXT_COLOR, COUNT_TEXT_THICKNESS)
    cv2.putText(frame, f"Males: {male_count}", (PERSON_COUNT_TEXT_POSITION[0], person_count_position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, COUNT_TEXT_SCALE, COUNT_TEXT_COLOR, COUNT_TEXT_THICKNESS)
    cv2.putText(frame, f"Females: {female_count}", (PERSON_COUNT_TEXT_POSITION[0], person_count_position[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, COUNT_TEXT_SCALE, COUNT_TEXT_COLOR, COUNT_TEXT_THICKNESS)

    return frame, car_count, male_count, female_count, person_count

def main():
    args = parse_arguments()
    model, cap = initialize_components(args.video)
    gender_model = YOLO("gender_classifier.pt")

    if model is None or cap is None:
        return  # Exiting the function if model or cap is None
    
    car_count = 0
    person_count = 0 
    male_count = 0
    female_count = 0
    counted_ids = set() # To keep track of the ids of the objects that have been counted

 # If the mode is 'save' the video is saved to the output path
    if args.mode == 'save':
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, input_fps, (600, 400))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, car_count, male_count, female_count, person_count = process_frame(frame, model, LINE_POSITION, car_count, male_count, female_count, person_count, counted_ids, gender_model)
        
# If the mode is 'view' the video is displayed on the screen
        if args.mode == 'view':
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break
        elif args.mode == 'save':
            out.write(frame)

    cap.release()
    if args.mode == 'save':
        out.release()
        print(f"The video has been saved at '{OUTPUT_VIDEO_PATH}'")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()