# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors



max_lane_signal_time = 30           # Maximum time per lane
total_lane_signal_time = 0
total_time_added = 0                # Initialize object detection added time
start_lane_time = time.time()
next_lane_go = False                # Next lane condition

max_pedestrian_wait_time = 10       # Maximum pedestrian wait time before crossing 
start_pedestrian_wait_time = 0
pedestrian_count = 0
pedestrian_go = False               # Pedestrian condition
pedestrian_detected = False         # Condition for starting pedestrian wait time

track_history = defaultdict(list)

start_region = None
counting_regions = [
    {
        "name": "Pedestrian Region",
        "polygon": Polygon([(900, 700), (1200, 700), (1200, 500), (900, 500)]), # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 155, 100),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "Multiplier Region",
        "polygon": Polygon([(400, 650), (700, 650), (650, 50), (500, 50)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (100, 200, 255),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

def next_lane_signal(total_time_added):
    global total_lane_signal_time, next_lane_go, start_lane_time
    
    
    total_lane_signal_time = total_time_added
    print(f"total_lane_signal_time: {total_lane_signal_time}")
    


    lane_elapsed_time = time.time() - start_lane_time
    print(f"elapsed_lane_time: {lane_elapsed_time}")
        
    if (max_lane_signal_time or total_lane_signal_time) <= lane_elapsed_time:
        next_lane_go = True
        print("Next lane signal: Go")
    else:
        print("Next lane signal: Stop")


def pedestrian_signal(pedestrian_count):
    global start_pedestrian_wait_time, pedestrian_go, pedestrian_detected
    
    if not pedestrian_detected:
        if pedestrian_count >= 3:
            start_pedestrian_wait_time = time.time()  # Start the pedestrian wait time counter
            print("Pedestrian wait time started")
            pedestrian_detected = True
        else:
            print("Pedestrian signal: Stop")
    else:
        pedestrian_elapsed_time = time.time() - start_pedestrian_wait_time
        print(f"elapsed_pedestrian_time: {pedestrian_elapsed_time}")
        
        if max_pedestrian_wait_time <= pedestrian_elapsed_time or pedestrian_count > 7:
            pedestrian_go = True
            print("Pedestrian signal: Go")
        else:
            print("Pedestrian signal: Stop")


def mouse_callback(event, x, y, flags, param):

    global start_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                start_region = region
                start_region["dragging"] = True
                start_region["offset_x"] = x
                start_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if start_region is not None and start_region["dragging"]:
            dx = x - start_region["offset_x"]
            dy = y - start_region["offset_y"]
            start_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in start_region["polygon"].exterior.coords]
            )
            start_region["offset_x"] = x
            start_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if start_region is not None and start_region["dragging"]:
            start_region["dragging"] = False


def run(
    source=None,
    device="cpu",
    hide_img=False,
    line_thickness=2,
    track_thickness=2,
    region_thickness=4,
):
    
    global total_lane_signal_time, next_lane_go, pedestrian_go, start_pedestrian_wait_time, total_time_added, pedestrian_count, start_lane_time
    
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO("runs/detect/train3/weights/best.pt")
    model.to("cuda") if device == "0" else model.to("cpu")
    
    model_cls = YOLO("runs/classify/train2/weights/best.pt")
    model_cls.to("cuda") if device == "0" else model_cls.to("cpu")
    
    # Extract classes names
    names = model.model.names
    names_cls = model_cls.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=None)
        results_cls = model_cls(frame)
        
        
        # Classification results
        for r in results_cls:
            # Convert the tensor to a numpy array
            top5_conf_np = r.probs.top5conf.numpy()
            
            # Get the top 5 class names
            top5_labels = [names_cls[i] for i in r.probs.top5]

            # Display class names and probabilities
            y_offset = 50
            for label, prob in zip(top5_labels, top5_conf_np):
                text = f"{label}: {prob:.2f}"
                cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y_offset += 30
            

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            
            classes_with_multiplier = {'bicycle': 0, 'bus': 0, 'car': 0, 'motorcycle': 0, 'truck': 0}

            for box, track_id, cls in zip(boxes, track_ids, clss):
                
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                
                # Check if detection inside rectangle region
                for region in counting_regions:
                    if region["name"] == "Multiplier Region" and region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        # Apply multipliers based on class
                        if cls == 0:  # bicycle
                            classes_with_multiplier['bicycle'] += 2
                        elif cls == 1:  # bus
                            classes_with_multiplier['bus'] += 5
                        elif cls == 2:  # car
                            classes_with_multiplier['car'] += 3
                        elif cls == 3:  # motorcycle
                            classes_with_multiplier['motorcycle'] += 2
                        elif cls == 5:  # truck
                            classes_with_multiplier['truck'] += 7
                
                # Check if detection inside polygon region 
                for region in counting_regions:
                    if region["name"] == "Pedestrian Region" and cls == 4 and region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1


                total_time_added = sum(classes_with_multiplier.values())

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = ""
            if region["name"] == "Pedestrian Region":
                if region["counts"] > 0:
                    region_label = f"Person Count: {region['counts']}"
                    pedestrian_count = region['counts']
                else:
                    region_label = "Person Count: 0"
            else:
                region_label = f"Time Added: {total_time_added:.2f}"

            region_color = region["region_color"]
            region_text_color = region["text_color"]


            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if not hide_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Optimized Traffic Light Inference Mode")
                cv2.setMouseCallback("Optimized Traffic Light Inference Mode", mouse_callback)
            cv2.imshow("Optimized Traffic Light Inference Mode", frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        next_lane_signal(total_time_added)
        pedestrian_signal(pedestrian_count)

    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--hide-img", action="store_true", help="hide results")


    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


# python actuator.py --source "traffic-view/AUP-1.mp4"

# {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'truck'}