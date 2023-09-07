import datetime
from ultralytics import YOLO
import cv2
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


# define some constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
# initialize the video capture object
video_cap = cv2.VideoCapture("sea_60.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "pretrained/day/detections_day/sea60_detections.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
count=0
ob_list=[]
t_list=[]
#model = YOLO("best.pt")
while True:
    # start time to compute the fps
    a=0
    b=0
    c=0
    d=0
    start = datetime.datetime.now()
    
    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]
    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        class_id = int(data[5])
        if class_id == 1:
           name = '1'
           a=a+1
        elif class_id == 0:
           name = '0'
           b=b+1
        else:
           name = 'Other'
           c=c+1
        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        d=a+b+c
        
        print("object count by frame=",d)
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        cv2.putText(frame, name, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        print(xmin, ymin, xmax, ymax)
        
    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print("total=", total)
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
    t_list.append(total*1000)
    count= count + 1
    ob_list.append(d)
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    #cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.putText(frame, str(d), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    # show the frame to our screen
    # cv2_imshow("Frame", frame)
    cv2.imshow("Frame",frame)
    writer.write(frame)
    filepath="pretrained/day/detections_day/frame"+ str(count)+".jpg"
    print(filepath)
    cv2.imwrite(filepath,frame)
    if cv2.waitKey(1) == ord("q"):
        break
    print("Frames count=", count)
    print("list", t_list)
    print("obj", ob_list)
video_cap.release()
writer.release()
cv2.destroyAllWindows()