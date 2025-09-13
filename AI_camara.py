import cv2 
import numpy as np
from ultralytics import YOLO
import time 
import json
import sqlite3
from datetime import datetime
import os

# CONFIGURATION FILE

with open("configure.json", "r") as f:
    config = json.load(f)

# video_path = config["rtsp stream"]
# video_path = config["video"]
video_path = int(config["front camera"])

"""===================================================================================="""

""" D A T A     B A S E     C O N N E C T I O N """
# Connect to DB (creates file if not exists)
conn = sqlite3.connect("vehicle_alert_logs.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS alerts (
    timestamp TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,
    vehicle_id INTEGER NOT NULL,
    event TEXT NOT NULL
    )
''')

conn.commit()

def log_detection(vehicle_type, vehicle_id, event):
    cursor.execute('''
    INSERT INTO alerts (timestamp, vehicle_type, vehicle_id, event)
    VALUES (?, ?, ?, ?)
    ''', (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        vehicle_type,
        vehicle_id,
        event
    ))
    conn.commit()


"""=============================================================================================="""

""" D R A W     Z O N E S """


zones = []
current_zone = []
drawing = False
preview_point = None

def draw_zones(event, x, y, flags, param):
    global current_zone, drawing, zones, preview_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_zone.append([x, y])  # Start a new point
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        preview_point = [x, y]  # Update preview point while dragging
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            preview_point = None  # Clear preview after commit
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_zone) >= 3:

            zones.append(current_zone.copy())
            current_zone = []
            drawing = False
            preview_point = None
        else:
            current_zone=[]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Get video properties

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(f"video:{video_path}")
print(f"Width: {width}\t Height: {height}\t FPS: {fps}")


# Check first frame

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()


# Generate filename for zones based on video name

video_basename = os.path.splitext(os.path.basename(str(video_path)))[0]
video_basename = video_basename.replace(" ", "_")
zones_path = f"zones_{video_basename}.json"


# Check for existing zones"

use_existing_zones = False
if os.path.exists(zones_path):
    with open(zones_path, "r") as f:
        data = json.load(f)
        if len(data["zones"])>0 and data["resolution"] == [width, height]:
            zones = data["zones"]

            # Display first frame with existing zones
            cv2.namedWindow("Existing Zones", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Existing Zones", width, height)
            temp_frame = frame.copy()
            if temp_frame.size == 0:
                print("Error: temp_frame is empty!")
                exit()
            for zone in zones:
                cv2.polylines(temp_frame, [np.array(zone, np.int32)], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            while True:
                cv2.imshow("Existing Zones", temp_frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                response = input("Do you want to reuse these zones? (y/n): ").strip().lower()
                if response in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'.")
            cv2.destroyWindow("Existing Zones")
            if response == 'y':
                use_existing_zones = True
            else:
                zones = []
        else:
            print("Warning: No zones or Resolution mismatch with saved zones. Redrawing zones.")


# Allow user to draw lane zones if none exist or they choose to redraw
if not use_existing_zones:
    cv2.namedWindow("Define Zones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define Zones", width, height)
    cv2.setMouseCallback("Define Zones", draw_zones)
    print("\n\t Zones are automatically named in the order they are drawn — the first zone is labeled Zone 1, the second Zone 2, and so on\n\t Make sure to take minimum 3 points to create zones." \
    "\n\t Left-click to add points.\n\t Press 'c' to remove the last point.\n\t Right-click to finish a zone." \
    "\n\t Press 'u' to undo the last zone.\n\t Press 'q' to finish/quit.")

    while True:
        temp_frame = frame.copy()
        for zone in zones:
            cv2.polylines(temp_frame, [np.array(zone, np.int32)], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        if current_zone:
            cv2.polylines(temp_frame, [np.array(current_zone, np.int32)], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            if preview_point and len(current_zone) > 0:
                cv2.line(temp_frame, tuple(current_zone[-1]), tuple(preview_point), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.imshow("Define Zones", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u') and len(zones) > 0:
            zones.pop()
        if key == ord('c') and len(current_zone) > 0:
            current_zone.pop()
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

# Save lane zones to JSON
with open(zones_path, "w") as f:
    json.dump({"resolution": [width, height], "zones": zones}, f)

# Generate unique filename for output video
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"output_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not initialize video writer.")
    cap.release()
    exit()



""" ==============================================================================================="""

""" R U N     Y O L O     M O D E L """


def test_point(bbox):
    x1, y1, x2, y2, track_id, conf, class_id = bbox
    #class_id = int(class_id)

    bottom_center = ( (x1+x2)/2 , y2 )
    center = ( (x1+x2)/2 , (y1+y2)/2 )
    bottom_right = (x2,y2)
    return (bottom_center, center, bottom_right)

def main():

    # Open input video
    #cap = cv2.VideoCapture(str(config["rtsp stream"]+"?fps="+config["IN_FPS"]))
    cap = cv2.VideoCapture(video_path)
    print(video_path.type())
    print(f"Loading video: {video_path}")
    if not cap.isOpened():
        print("Error opening input video.")
        return

    # Get video info
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    print(f"Width: {width}\t Height: {height}\t FPS: {fps}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error opening output video.")
        return

    # run yolo model
    print(f"Loading {config["yolo model"]} model...")
    model = YOLO(config["yolo model"])
    
    # start Video process
    print("Start Video Processing...")
    count = 0
    fps_value = "--"
    prev_time = 0
    while True:

        count += 1
        ret, frame = cap.read()
        if not ret:
            print("No more frames left in Video.")
            break

        for i in zones:
            zone_np = np.array(i, np.int32)
            cv2.polylines(frame, [zone_np], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # Track objects
        results = model.track(frame ,persist=True, conf = config["conf"], iou = config["iou"], imgsz=config["imgsz"], classes=config["classes"], verbose=False)[0]   # tracker = BOTSORT

        # check IF frame has any detaction and track ID
        if len(results.boxes.xyxy)>0 and results.boxes.id is not None:

            bboxs = results.boxes.data.tolist()
            
            for bbox in bboxs:
                # select point for selection zone test i.e.
                check_point = test_point(bbox)    #Bottom center, Center, Bottom right
                check_point = check_point[config["box_center"]]

                # check if bbox in selection zone
                for i, zone in enumerate(zones):   
                    zone_np = np.array(np.array(zone), np.int32)                 
                    if cv2.pointPolygonTest(zone_np, check_point, False) >= 0:

                        # save coordinates
                        x1, y1, x2, y2, track_id, conf, class_id = bbox
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        track_id = int(track_id)
                        class_id = int(class_id)
                        type = model.names[class_id]
                        
                        label = f"{type} ID: {track_id}"
                        
                        # draw boxes
                        (text_width, text_height), baseline = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 0, 0), -1)
                        draw_point = tuple(map(int, check_point))
                        cv2.circle(frame, draw_point, 3, (0,0,255), -1)
                        # Draw white text on top of the background
                        cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        fps_value = f"{fps:.1f}"
        cv2.putText(frame, f"FPS: {fps_value}", (width - 115, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
        # Save Output
        out.write(frame)        
        
        # Show output
        if True:
            cv2.imshow("Vehicle Detection and Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()


if __name__ == "__main__":

    main()
