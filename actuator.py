from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import pyfirmata
import time

board = pyfirmata.Arduino('/dev/tty.usbmodem1101')
print("Communication Successfully started")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("traffic-view/AUP-1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(20, 400), (1080, 400)]  # line or region points
classes_to_count = [0, 1, 2, 3, 5, 7] 

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                reg_pts=line_points,
                classes_names=model.names,
                draw_tracks=True,
                line_thickness=2)

while True:
    
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False,
                            classes=classes_to_count)

        im0 = counter.start_counting(im0, tracks)
        #video_writer.write(im0)
        
        board.digital[12].write(1)
        board.digital[8].write(1)
        board.digital[7].write(1)
        time.sleep(1)
        board.digital[12].write(0)
        board.digital[8].write(0)
        board.digital[7].write(0)
        time.sleep(1)

    cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()