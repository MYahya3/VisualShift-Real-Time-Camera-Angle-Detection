import cv2
import glob

source_folder = "D:\MLOps\CDS\Camera_angle_change_Realtime\RAS/raw_data\RAS_104/*.mp4"
vid = glob.glob(source_folder)
# Write Video
out = cv2.VideoWriter("D:\MLOps\CDS\Camera_angle_change_Realtime\RAS/raw_data/RAS_104_Drift.mp4", cv2.VideoWriter.fourcc(*'.mp4'), 5.0, (640, 480))

# To stack and write videos as 1 video
for v in vid:
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (640, 360))
        # print(frame.shape[1], frame.shape[0])
        out.write(frame)
        cv2.waitKey(1)
