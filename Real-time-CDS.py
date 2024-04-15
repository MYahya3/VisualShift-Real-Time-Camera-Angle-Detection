import os
import time
import cv2
import numpy as np
from helper_functions import DrawOpac, calculate_iou, drawError
import json

def light_off(frame, brightness_thre):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    center_x, center_y = 0.65, 1.0  # Assuming the center
    # Define regions
    center_x1 = int(w * (1 - center_x) // 2)
    center_x2 = int(w * (1 + center_x) // 2)
    center_y1 = int(h * (1 - center_y) // 2)
    center_y2 = int(h * (1 + center_y) // 2)

    center_region = gray[center_y1:center_y2, center_x1:center_x2]
    outer_region = gray.copy()
    outer_region[center_y1:center_y2, center_x1:center_x2] = 0

    # Calculate mean intensities
    center_intensity = np.mean(center_region)
    outer_intensity = np.mean(outer_region)
    # print(center_intensity, outer_intensity)
    return outer_intensity < brightness_thre or (
            center_intensity < brightness_thre and outer_intensity < brightness_thre)

### To Get ROIS from Reference Image ###
def getROIS(image, roi_list=None):
    if os.path.exists(image):
        gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # Define the region of interest in the reference image
        x, y, w, h = roi_list[0], roi_list[1], roi_list[2], roi_list[3]
        roiTemplate = gray_image[y:y + h, x:x + w]  # adjust the coordinates and size to your specific ROI
        return roiTemplate
    else:
        pass

#### Match Template Fn Results #####
def matchTemplate(image, ref_image, method):
    result = cv2.matchTemplate(image, ref_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < 0:
        max_val = 0
    pred_loc = [max_loc[0], max_loc[1], ref_image.shape[1], ref_image.shape[0]]
    return max_val, pred_loc

#### Camera Angle Evaluation Function ####
def driftFinder(frame, ref_image_pth, ROisImg_List,IOU_THRES,ROI_coords, ErrorCount, alertTimer=60):

        ## To save IOI results , Similarity Results, Pred_Loc for all Rois ##
        IOU_List, Similarity_List, Pred_Loc = [], [], []

        ref_image = cv2.imread(ref_image_pth, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the video frame to grayscale
        # For Loop to get Results e.g IOU List, Similarity and Pred Loc list
        for coord,img in zip(ROI_coords, ROisImg_List):
            max_val, pred_loc = matchTemplate(gray_frame, ref_image= img, method=cv2.TM_CCOEFF_NORMED)
            BOX_IOU = round(calculate_iou(coord, pred_loc) * 100)
            Similarity_List.append(round(max_val * 100))
            IOU_List.append(BOX_IOU)
            Pred_Loc.append(pred_loc)

        # Max value index in list
        max_value_idx = np.argmax(IOU_List)

                                                    ### Main Logic ###
        if Similarity_List[max_value_idx] > 20:
            if IOU_List[max_value_idx] < IOU_THRES :
                ErrorCount += 1
            else:
                ErrorCount = 0
        if alertTimer + 10 > ErrorCount > alertTimer:
            dir = os.path.dirname(ref_image_pth)
            cv2.imwrite(f"{dir}/old_refImg.png", ref_image)
            ## Create Red Box in Top-left corner ##
            if IOU_List[max_value_idx] > 5:
                frame = drawError(image=frame, actualROI=ROI_coords[max_value_idx], predROI=Pred_Loc[max_value_idx])

            frame = DrawOpac(frame, alpha=0.8, bbox=[0, 0, 220, 35], color=(114, 114, 238), DrawShape="rect")
            ref_image = drawError(image=ref_image, actualROI=ROI_coords[max_value_idx], predROI=ROI_coords[max_value_idx])
            cv2.putText(frame, f"Cam-Angle: {IOU_List[max_value_idx]}%  [{IOU_THRES}%]", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            driftImg = np.hstack([ref_image, frame])
            # Remove Ref_image if drift detected #
            cv2.imwrite(f"{dir}/Drift.png", driftImg)
            os.remove(ref_image_pth)
        else:
            frame = DrawOpac(frame, alpha=0.8, bbox=[0, 0, 220, 35], color=(200, 200, 200), DrawShape="rect")
            cv2.putText(frame, f"Cam-Angle: {IOU_List[max_value_idx]}%  [{IOU_THRES}%]", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame, ErrorCount

                                              ###### Main Loop ######
def main(json_file, source_video, ref_image_pth, Thres, refAutoSelection = False):

    ## To write Video
    # out = cv2.VideoWriter("D:\MLOps\CDS\Camera_angle_change_Realtime\RAS\RAS_104/RAS_104_output.mp4", cv2.VideoWriter.fourcc(*'.mp4'), 5.0, (640, 480))

    # GET Variables from JSON FILE
    if os.path.exists(json_file):
        with open(json_file, "r") as read_it:
            data = json.load(read_it)
            roi_list = data["rects"]

        # Initialize Video
        cap = cv2.VideoCapture(source_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        #### Write Output Video ####
        if not cap.isOpened():
            print(f"Error: Could not open {source_video}.")
            exit()

        Count = 0  # For Ref Image auto selection after 10 frames
        ErrorCount = 0

        # Get list of Ref Roi templates for each roi defined
        roiGrayImgs = [(getROIS(image=ref_image_pth, roi_list=roi)) for roi in roi_list]

        #### While ### Loop ####
        while True:
            starttm = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            # print(frame.shape[1], frame.shape[0])
            # To auto select Reference Image
            if refAutoSelection == True and Count == 20:
                cv2.imwrite(ref_image_pth, frame)
                roiGrayImgs = [(getROIS(image=ref_image_pth, roi_list=roi)) for roi in roi_list]

            # Check if light-off
            if light_off(frame, 28):
                continue

            # Check if the video has reached the end
            image = frame
            Count += 1

            if os.path.exists(ref_image_pth):
                image, ErrorCount = driftFinder(frame=image, ref_image_pth=ref_image_pth, ROisImg_List=roiGrayImgs, ROI_coords=roi_list,
                   IOU_THRES=Thres, ErrorCount=ErrorCount, alertTimer= int(3*fps))
                print(ErrorCount)
            else:
                pass
            cv2.imshow('Video-1', image)
            # out.write(image)
            endtime = time.time()
            elapsed = endtime - starttm
            # print(elapsed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Error: ROIs Json File not Found")
    # out.release()

if __name__ == "__main__":

    source = 1
    ref_image = "D:\MLOps\CDS\Camera_angle_change_Realtime\office_test/ref_image.png"
    IOU_THRES = 90
    json_file = "D:\MLOps\CDS\Camera_angle_change_Realtime\office_test/roi_list1280x720.json"

    main(json_file=json_file, source_video=source, ref_image_pth=ref_image, Thres=IOU_THRES, refAutoSelection=False)