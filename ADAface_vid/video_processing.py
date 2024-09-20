#!/usr/bin/env python3

# import cv2

# from ultralytics import YOLO

# from deep_sort_realtime.deepsort_tracker import DeepSort

# import numpy as np



# def process_video(yolo_model, video_path, confidence_threshold=0.7):

#     cap = cv2.VideoCapture(video_path)



#     if not cap.isOpened():

#         raise ValueError("Failed to open video file. Check the file path.")



#     frame_count = 0  # To keep track of frame numbers



#     # Initialize DeepSort tracker with adjusted parameters

#     tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)



#     while True:

#         success, frame = cap.read()

#         if not success:

#             break



#         # Skip frames based on the frame count to reduce processing load

#         if frame_count % 1 != 0:

#             frame_count += 1

#             continue



#         # Detect faces using YOLOv8 with the confidence threshold

#         results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)



#         # Prepare detections for DeepSort

#         detections = []

#         for result in results:

#             boxes = result.boxes.xyxy.cpu().numpy()

#             scores = result.boxes.conf.cpu().numpy()

#             class_ids = result.boxes.cls.cpu().numpy()



#             if boxes is None or len(boxes) == 0:

#                 print("No faces detected.")

#                 continue



#             for box, score, class_id in zip(boxes, scores, class_ids):

#                 # if score < confidence_threshold:

#                 #     continue  # Skip boxes with confidence below the threshold



#                 if int(class_id) != 0:  # Assuming face class ID is 0

#                     continue



#                 x1, y1, x2, y2 = map(int, box[:4])

#                 detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])



#         # Update DeepSort tracker with detections

#         tracks = tracker.update_tracks(detections, frame=frame)



#         # Loop over the tracks and assign IDs

#         for track in tracks:

#             if not track.is_confirmed() or track.time_since_update > 0:

#                 continue  # Only consider tracks that have been updated in the current frame



#             track_id = track.track_id

#             ltrb = track.to_ltrb()



#             xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])



#             # Draw the bounding box around the face

#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)



#             # Label the bounding box with the track ID

#             cv2.putText(frame, f"ID {track_id}", (xmin + 5, ymin - 8),

#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



#         # Display the resulting frame

#         cv2.imshow("Frame", frame)



#         # Save the frame as an image file for remote viewing (optional)

#         # cv2.imwrite(f"/path/to/output/frame_{frame_count}.jpg", frame)



#         # Increment frame count

#         frame_count += 1



#         # Exit on 'q' key press

#         if cv2.waitKey(1) & 0xFF == ord('q'):

#             break



#     cap.release()

#     cv2.destroyAllWindows()



# if __name__ == "__main__":

#     model_path = 'yolov8m_200e.pt'  # Adjust this path as needed

#     yolo_model = YOLO(model_path)



#     video_path = 'FacingCam01.mp4'  # Adjust this path as needed

#     process_video(yolo_model, video_path)










# # #!/usr/bin/env python3

# import cv2
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import numpy as np
# import os

# def is_frontal_face(box):
#     """
#     Determine if the face is frontal based on the aspect ratio of the bounding box.
#     Args:
#         box: Bounding box (x1, y1, x2, y2) of the detected face.
#     Returns:
#         bool: True if the face is frontal, False otherwise.
#     """
#     x1, y1, x2, y2 = box
#     width = x2 - x1
#     height = y2 - y1
#     aspect_ratio = width / height

#     # Check if the face has an aspect ratio close to 1:1
#     return 0.8 <= aspect_ratio <= 1.2

# def process_video(yolo_model, video_path, confidence_threshold=0.7, output_dir="output"):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise ValueError("Failed to open video file. Check the file path.")

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     frame_count = 0
#     tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)
#     saved_ids = {}

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         if frame_count % 1 != 0:
#             frame_count += 1
#             continue

#         results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)

#         detections = []
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy()
#             scores = result.boxes.conf.cpu().numpy()
#             class_ids = result.boxes.cls.cpu().numpy()

#             if boxes is None or len(boxes) == 0:
#                 continue

#             for box, score, class_id in zip(boxes, scores, class_ids):
#                 if int(class_id) != 0:  # Assuming face class ID is 0
#                     continue

#                 x1, y1, x2, y2 = map(int, box[:4])
#                 detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

#         tracks = tracker.update_tracks(detections, frame=frame)

#         for track in tracks:
#             if not track.is_confirmed() or track.time_since_update > 0:
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

#             track_img = frame[ymin:ymax, xmin:xmax]

#             # Check if we already have a frontal face for this track ID
#             if track_id in saved_ids:
#                 continue

#             if is_frontal_face((xmin, ymin, xmax, ymax)):
#                 # Save frontal face
#                 save_path = os.path.join(output_dir, f"track_{track_id}_frontal.jpg")
#                 cv2.imwrite(save_path, track_img)
#                 saved_ids[track_id] = "frontal"
#             else:
#                 # Save side face if no frontal face exists
#                 save_path = os.path.join(output_dir, f"track_{track_id}_side.jpg")
#                 cv2.imwrite(save_path, track_img)
#                 saved_ids[track_id] = "side"

#         frame_count += 1

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model_path = 'yolov8m_200e.pt'  # Adjust this path as needed
#     yolo_model = YOLO(model_path)

#     video_path = 'FacingCam01.mp4'  # Adjust this path as needed
#     process_video(yolo_model, video_path)



# import cv2
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import numpy as np
# import os

# def is_frontal_face(box):
#     """
#     Determine if the face is frontal based on the aspect ratio of the bounding box.
#     Args:
#         box: Bounding box (x1, y1, x2, y2) of the detected face.
#     Returns:
#         bool: True if the face is frontal, False otherwise.
#     """
#     x1, y1, x2, y2 = box
#     width = x2 - x1
#     height = y2 - y1
#     aspect_ratio = width / height

#     # Check if the face has an aspect ratio close to 1:1
#     return 0.8 <= aspect_ratio <= 1.2

# def calculate_sharpness(image):
#     """
#     Calculate the sharpness of the image using the variance of the Laplacian.
#     Args:
#         image: Input image (BGR).
#     Returns:
#         float: Sharpness score of the image.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#     variance = laplacian.var()
#     return variance

# def process_video(yolo_model, video_path, confidence_threshold=0.7, output_dir="output", side_faces_dir="side_faces"):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise ValueError("Failed to open video file. Check the file path.")

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     if not os.path.exists(side_faces_dir):
#         os.makedirs(side_faces_dir)

#     frame_count = 0
#     tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)
#     saved_ids = {}

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         if frame_count % 1 != 0:
#             frame_count += 1
#             continue

#         results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)

#         detections = []
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy()
#             scores = result.boxes.conf.cpu().numpy()
#             class_ids = result.boxes.cls.cpu().numpy()

#             if boxes is None or len(boxes) == 0:
#                 continue

#             for box, score, class_id in zip(boxes, scores, class_ids):
#                 if int(class_id) != 0:  # Assuming face class ID is 0
#                     continue

#                 x1, y1, x2, y2 = map(int, box[:4])
#                 detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

#         tracks = tracker.update_tracks(detections, frame=frame)

#         for track in tracks:
#             if not track.is_confirmed() or track.time_since_update > 0:
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

#             track_img = frame[ymin:ymax, xmin:xmax]

#             # Check if we already have a frontal face for this track ID
#             if track_id in saved_ids:
#                 current_best_sharpness, saved_face_type, _ = saved_ids[track_id]
#                 current_sharpness = calculate_sharpness(track_img)
#                 if is_frontal_face((xmin, ymin, xmax, ymax)):
#                     # If the new face is frontal and the saved image is not frontal or is less sharp
#                     if saved_face_type != "frontal" or current_sharpness > current_best_sharpness:
#                         save_path = os.path.join(output_dir, f"track_{track_id}_frontal.jpg")
#                         cv2.imwrite(save_path, track_img)
#                         saved_ids[track_id] = (current_sharpness, "frontal", save_path)
#                 else:
#                     # If the new face is side and the saved image is not side or is less sharp
#                     if saved_face_type != "side" or current_sharpness > current_best_sharpness:
#                         save_path = os.path.join(side_faces_dir, f"track_{track_id}_side.jpg")
#                         cv2.imwrite(save_path, track_img)
#                         saved_ids[track_id] = (current_sharpness, "side", save_path)
#             else:
#                 if is_frontal_face((xmin, ymin, xmax, ymax)):
#                     save_path = os.path.join(output_dir, f"track_{track_id}_frontal.jpg")
#                     saved_face_type = "frontal"
#                 else:
#                     save_path = os.path.join(side_faces_dir, f"track_{track_id}_side.jpg")
#                     saved_face_type = "side"

#                 current_sharpness = calculate_sharpness(track_img)
#                 cv2.imwrite(save_path, track_img)
#                 saved_ids[track_id] = (current_sharpness, saved_face_type, save_path)

#         frame_count += 1

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model_path = 'yolov8m_200e.pt'  # Adjust this path as needed
#     yolo_model = YOLO(model_path)

#     video_path = 'FacingCam01.mp4'  # Adjust this path as needed
#     process_video(yolo_model, video_path)





# import cv2
# import dlib
# import os
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Load the dlib face landmark detector
# detector = dlib.get_frontal_face_detector()
# predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Path to the shape predictor model file
# predictor = dlib.shape_predictor(predictor_path)  # Initialize shape predictor

# def is_frontal_face(landmarks, min_landmarks=5):
#     """
#     Check if the face is frontal based on the number of key landmarks detected.
#     """
#     return len(landmarks.parts()) >= min_landmarks

# def process_video(yolo_model, video_path, output_dir='output', confidence_threshold=0.7):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("Failed to open video file. Check the file path.")

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Initialize DeepSort tracker with adjusted parameters
#     tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)

#     # Dictionary to keep track of the best face for each unique ID
#     first_frontal_faces = {}

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Detect faces using YOLOv8 with the confidence threshold
#         results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)

#         # Prepare detections for DeepSort
#         detections = []
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy()
#             scores = result.boxes.conf.cpu().numpy()
#             class_ids = result.boxes.cls.cpu().numpy()

#             for box, score, class_id in zip(boxes, scores, class_ids):
#                 if int(class_id) != 0:  # Assuming face class ID is 0
#                     continue

#                 x1, y1, x2, y2 = map(int, box[:4])
#                 detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

#         # Update DeepSort tracker with detections
#         tracks = tracker.update_tracks(detections, frame=frame)

#         # Loop over the tracks and assign IDs
#         for track in tracks:
#             if not track.is_confirmed() or track.time_since_update > 0:
#                 continue  # Only consider tracks that have been updated in the current frame

#             track_id = track.track_id
#             ltrb = track.to_ltrb()

#             xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

#             # Extract face from frame
#             face_image = frame[ymin:ymax, xmin:xmax]

#             # Detect landmarks for the extracted face
#             dlib_rect = dlib.rectangle(xmin, ymin, xmax, ymax)
#             landmarks = predictor(frame, dlib_rect)
            
#             # Check if the face is frontal and has at least 5 landmarks
#             if is_frontal_face(landmarks):
#                 # Store the first valid frontal face for each unique ID
#                 if track_id not in first_frontal_faces:
#                     first_frontal_faces[track_id] = face_image

#     cap.release()

#     # Save the first frontal face for each unique ID
#     for track_id, face_image in first_frontal_faces.items():
#         cv2.imwrite(os.path.join(output_dir, f"first_frontal_face_ID_{track_id}.jpg"), face_image)

# # Load the YOLOv8 model
# model_path = 'yolov8m_200e.pt'
# yolo_model = YOLO(model_path)

# # Define the video path and output directory
# video_path = 'FacingCam01.mp4'
# output_dir = 'output_faces/'

# # Process the video
# process_video(yolo_model, video_path, output_dir)





import cv2
import dlib
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the dlib face landmark detector
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Path to the shape predictor model file
predictor = dlib.shape_predictor(predictor_path)  # Initialize shape predictor

def is_frontal_face(landmarks, min_landmarks=5):
    """
    Check if the face is frontal based on the number of key landmarks detected.
    """
    return len(landmarks.parts()) >= min_landmarks

def image_sharpness(image):
    """
    Calculate the sharpness of an image using the variance of the Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video(yolo_model, video_path, output_dir='output', confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file. Check the file path.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize DeepSort tracker with adjusted parameters
    tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)

    # Dictionary to keep track of the best face for each unique ID
    best_faces = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect faces using YOLOv8 with the confidence threshold
        results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)

        # Prepare detections for DeepSort
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                if int(class_id) != 0:  # Assuming face class ID is 0
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

        # Update DeepSort tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Loop over the tracks and assign IDs
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue  # Only consider tracks that have been updated in the current frame

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Extract face from frame
            face_image = frame[ymin:ymax, xmin:xmax]

            # Detect landmarks for the extracted face
            dlib_rect = dlib.rectangle(xmin, ymin, xmax, ymax)
            landmarks = predictor(frame, dlib_rect)
            
            # Check if the face is frontal and has at least 5 landmarks
            if is_frontal_face(landmarks):
                # Calculate the sharpness of the face image
                sharpness = image_sharpness(face_image)
                
                # Store the clearest image for each unique ID
                if track_id not in best_faces or sharpness > best_faces[track_id]["sharpness"]:
                    best_faces[track_id] = {
                        "image": face_image,
                        "sharpness": sharpness
                    }

    cap.release()

    # Save the clearest face for each unique ID
    for track_id, data in best_faces.items():
        face_image = data["image"]
        cv2.imwrite(os.path.join(output_dir, f"clearest_face_ID_{track_id}.jpg"), face_image)

# Load the YOLOv8 model
model_path = 'yolov8m_200e.pt'
yolo_model = YOLO(model_path)

# Define the video path and output directory
video_path = 'FacingCam01.mp4'
output_dir = 'output_faces/'

# Process the video
process_video(yolo_model, video_path, output_dir)

