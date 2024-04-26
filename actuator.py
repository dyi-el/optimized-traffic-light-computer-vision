# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
from collections import defaultdict
from pathlib import Path
import time
import pyfirmata
import asyncio

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


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

async def next_lane_go():
    global board, next_source

    board.digital[12].write(1)
    await asyncio.sleep(1)
    board.digital[12].write(0)
    next_source = True
 
async def pedestrian_go():       
    global board, next_source
    
    board.digital[8].write(1)
    await asyncio.sleep(1)
    board.digital[8].write(0)
    
    await asyncio.sleep(10)
    board.digital[8].write(1)
    await asyncio.sleep(0.5)
    board.digital[8].write(0)
    
    board.digital[12].write(1)
    await asyncio.sleep(1)
    board.digital[12].write(0)
    next_source = True
        
async def warning_go():
    global board, next_source
    
    board.digital[7].write(1)
    await asyncio.sleep(1)
    board.digital[7].write(0)
    
    await asyncio.sleep(10)
    
    board.digital[7].write(1)
    await asyncio.sleep(1)
    board.digital[7].write(0)
    
    next_source = True


def reset_variables():
    """Reset or reinitialize variables."""
    global max_lane_signal_time, total_lane_signal_time, start_lane_time
    global max_pedestrian_wait_time, start_pedestrian_wait_time, pedestrian_count
    global pedestrian_detected, detect_added, classify_added, next_source
    
    max_lane_signal_time = 80
    total_lane_signal_time = 0
    start_lane_time = time.time()


    max_pedestrian_wait_time = 10
    start_pedestrian_wait_time = 0
    pedestrian_count = 0
    pedestrian_detected = False

    detect_added = 0
    classify_added = 10
    
    next_source = False
    

def next_lane_signal(detect_added, classify_added, results_cls, top5_labels, top5_conf_np):
    global total_lane_signal_time, start_lane_time
    
    
    for r in results_cls:
        # Iterate over each class in the classification results
        for label, prob in zip(top5_labels, top5_conf_np):
            # Apply conditions based on class probabilities
            if label == "sparse_traffic" and prob > 0.80:
                classify_added *= 0.5
            elif label == "dense_traffic" and prob > 0.30:
                classify_added *= 1.2
            elif label == "fire" and prob > 0.60:
                #cv2.putText(frame,"WARNING", (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 140, 255), 2)
                #asyncio.run(warning_go())
                continue
            elif label == "accident" and prob > 0.60:
                #cv2.putText(frame,"WARNING", (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 140, 255), 2)
                #asyncio.run(warning_go())
                continue               
                
    total_lane_signal_time = detect_added + classify_added
    
    lane_elapsed_time = time.time() - start_lane_time
    
    
    if min(max_lane_signal_time, total_lane_signal_time) <= lane_elapsed_time:
        #cv2.putText(frame,"Next Lane", (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        asyncio.run(next_lane_go())
        return total_lane_signal_time, lane_elapsed_time
    else:
        return total_lane_signal_time, lane_elapsed_time
    
    

def pedestrian_signal(pedestrian_count):
    global start_pedestrian_wait_time, pedestrian_detected
    
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
            #cv2.putText(frame,"Pedestrian Go", (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            asyncio.run(pedestrian_go())


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
    board_port='/dev/tty.usbmodem1401',
    line_thickness=2,
    track_thickness=2,
    region_thickness=4,
):
    
    global total_lane_signal_time, start_pedestrian_wait_time, detect_added
    global classify_added, pedestrian_count, start_lane_time, board, next_source
    
    board = pyfirmata.Arduino(board_port)
    print("Communication Successfully started")
    
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO("runs/detect/train3/weights/best.pt")
    model.to("cuda") if device == "0" else model.to(device)
    
    model_cls = YOLO("runs/classify/train2/weights/best.pt")
    model_cls.to("cuda") if device == "0" else model_cls.to(device)
    

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
            top5_conf_np = r.probs.top5conf
            
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


                detect_added = sum(classes_with_multiplier.values())

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
                region_label = f"Time Added: {detect_added:.2f}"

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

        
        
        total_time, elapsed_time = next_lane_signal(detect_added, classify_added, results_cls, top5_labels, top5_conf_np)
        print("Total time:", total_time)
        print("Elapsed time:", elapsed_time)
        text_total = f"Total Optimized Time: {total_time:.2f}"
        text_elapsed = f"Elapsed Time: {elapsed_time:.2f}"
        
        cv2.putText(frame, text_total, (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, text_elapsed, (700, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        pedestrian_signal(pedestrian_count)
        
        
        if not hide_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Optimized Traffic Light Inference Mode")
                cv2.setMouseCallback("Optimized Traffic Light Inference Mode", mouse_callback)
            cv2.imshow("Optimized Traffic Light Inference Mode", frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if next_source:
            break
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cuda device, cpu or mps")
    parser.add_argument("--hide-img", action="store_true", help="hide results")
    parser.add_argument("--board-port", required=True, help="ls /dev/tty.*")


    return parser.parse_args()


def main(opt):
    """Main function."""
    # List of source paths
    source_list = [
        "traffic-view/Paseo-1.mp4",
        "traffic-view/Paseo-2.mp4",
        "traffic-view/Paseo-3.mp4",
        "traffic-view/Paseo-4.mp4"
    ]
    while True:
        for source_path in source_list:
            reset_variables() 
            opt.source = source_path
            run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


# python actuator.py --source "traffic-view/AUP-1.mp4" --device "mps"

# {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'truck'}