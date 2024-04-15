import cv2
import json
import os

# Global variables
rects = []
roi_count = 0

def save_to_json(data, json_filename):
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=4)

def load_from_json(json_filename):
    if not os.path.exists(json_filename):
        return {}
    with open(json_filename, "r") as f:
        content = f.read()
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {}
    return {}

def visaulize_rois(datadir,json_file, frame):
    with open(f"{datadir}/{json_file}", "r") as read_it:
        data = json.load(read_it)
        if len(data) > 0:
            roi_list = data["rects"]
            for roi in roi_list:
                print(roi)
                x, y, w, h = roi
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            print("No ROIs found in Json")
            exit()
    return frame

def roi_selection(frame):
    global points, rects, roi_count, poly_count

    while True:
        cv2.imshow('image', frame)
        key = cv2.waitKey(100) & 0xFF

        roi = cv2.selectROI("image", frame, False)
        if roi == (0, 0, 0, 0):
            break

        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_data = (x, y, w, h)
        rects.append(roi_data)
        roi_count += 1
        print(f"ROI_{roi_count}:", roi_data)

        if key == 27:  # Escape key or exit flag set
            break

def main(video_filename,json_filename, roiSelection = False, data_dir = None):

    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    cv2.namedWindow("Video Frame")
    while True:
        ret, frame = cap.read()
        dummy = frame.copy()
        if not ret:
            break
        # Check for key press
        key = cv2.waitKey(0) & 0xFF
        if roiSelection == True:
            # Display the frame
            cv2.imshow('Video Frame', frame)

            if key == 13:  # 13 is the ASCII code for the Enter key
                roi_selection(frame)
            if key & 0xFF == 13:
                cv2.imwrite(f"{data_dir}/ref_image.png", dummy)
                data = {}
                if rects:
                    data["rects"] = rects
                # Save data to the JSON file
                save_to_json(data, json_filename=f"{data_dir}/{json_filename}")
                break
        else:
            frame = visaulize_rois(datadir=data_dir,json_file=json_filename, frame=frame)
            cv2.imshow("Video Frame", frame)
            if key & 0xFF == 13:
                cv2.imwrite\
                    (f"{data_dir}/ref_image.png", dummy)
                break

    cap.release()

if __name__ == "__main__":
    video_filename = 1 # Provide the path to your video file

    main(video_filename,json_filename="roi_list1280x720.json", roiSelection=True, data_dir="D:/MLOps/CDS/Camera_angle_change_Realtime/office_test")
