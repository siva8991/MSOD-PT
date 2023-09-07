import threading
import datetime
import time
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
#from google.colab.patches import cv2_imshow


def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

"""def get_user_input():
    user_input = None

    def prompt_input():
        nonlocal user_input
        user_input = input("Enter your input: ")

    input_thread = threading.Thread(target=prompt_input)
    input_thread.start()
    input_thread.join(timeout=5)  # Set the timeout to 2 seconds

    if input_thread.is_alive():
        print("No input provided within 2 seconds. Returning back to the code.")
        # Perform actions when no input is received within the time limit
    else:
        # Process the user input
        print("User input:", user_input)
        # Continue with the rest of your code
    return user_input"""

#one=[]

"""def operations(val):
   one.append(val)
   print("operations:", one)
   return one"""

def readfile():
    file_path = "text.txt"  # Replace with the actual file path
    two=[]
    try:
        with open(file_path, "r") as file:
        # Read the contents of the file
            for line in file:
            # Append each line to the list
                two.append(line.strip())
        # Print the file contents
            print(two)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except IOError:
        print("An error occurred while reading the file.")
    return two
    
CONFIDENCE_THRESHOLD = 0.8
box = (255, 255, 0)
text = (255, 255, 255)
prior = (0, 0, 0)
# initialize the video capture object
video_cap = cv2.VideoCapture("sea_60.mp4")
#C:/Users/sivap/OneDrive/Desktop/deva/
# initialize the video writer object
writer = create_video_writer(video_cap, "pretrained/day/sea_60_tracking_0.8.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=20)

one=readfile()
print(one)
count=0
ob_list=[]
t_list=[]
track_list=[]
while True:
    start = datetime.datetime.now()
    a=0
    b=0
    c=0
    d=0
    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]
        #print("Confidence=",confidence)
        a=a+1
        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
        #print(results)
        print("object count=",a)
    ######################################
    # TRACKING
    ######################################
    track1=[]
    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        track1.append(track_id)
        ltrb = track.to_ltrb()
        #print(ltrb)
        print("Track Id=",track_id)
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        if track_id not in one:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), box, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text, 2)

        for va in one:
            if track_id == va:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), prior, 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), prior, 3)
                cv2.putText(frame, str(va), (xmin + 10, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, prior , 2)
        
       
        #print("object_track_list", track_list)        
    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    t_list.append(total*1000)
    count= count + 1
    ob_list.append(a)
    track_list.append(len(track1))
    print("length of the object tracking by frame=", len(track1))
    # calculate the frame per second and draw it on the frame
    #fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    #cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.putText(frame, str(a), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    one=readfile()
    print(one)
    
    """var = get_user_input()
    print("input of user:", var)
    if var is None:
        print("Don't store in one list")
    else:
        operations(var)"""
    print("one values=",one)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    #cv2_imshow(frame)
    filepath="pretrained/day/track_0.8/frame"+ str(count)+".jpg"
    print(filepath)
    cv2.imwrite(filepath,frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break
    print("object tracking count by frame=", track_list)
    print("Frames count=", count)
    print("time calculation=", t_list)
    #print("list", t_list)
    print("obj", ob_list)
video_cap.release()
writer.release()
cv2.destroyAllWindows()