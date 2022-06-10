import boto3
import os
import cv2
import numpy as np
import importlib.util
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import datetime

client = boto3.client('rekognition')

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("novice-342709-9a8f64515cff.json", scopes)
file = gspread.authorize(credentials) 
sheet = file.open("NPAws").get_worksheet(0)
                
threshold=0.5
GRAPH_NAME = 'model.tflite'
LABELMAP_NAME = 'label.txt'
vid_path = input("Enter video path -> ")

t = vid_path.split('/')
VIDEO_NAME = t[-1]

min_conf_threshold = float(threshold)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()
print('CWD_PATH:',CWD_PATH)
# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH, vid_path)
print('video path:::::', VIDEO_PATH)
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, GRAPH_NAME)
print('model file dir:::::',PATH_TO_CKPT)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, LABELMAP_NAME)
print('label file dir:::::',PATH_TO_LABELS)
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)       
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("VIDEO_NAME :::::::::::",VIDEO_NAME)                

Videos_Time = re.sub('\D', '', VIDEO_NAME) #remove all things exclude number
START_Videos_Time = str(Videos_Time[0:4]+'-'+Videos_Time[4:6]+'-'+Videos_Time[6:8]+' '
    +Videos_Time[8:10]+":"+Videos_Time[10:12]+":"+Videos_Time[12:14]) 
print("START_Videos_Time ::::::::::::",START_Videos_Time)

def convert(seconds):
    seconds = seconds % (24 * 3600)
    print("seconds ::::::::",seconds)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return seconds,minutes,hour

def getFrame(sec): 
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    return video.read() 
sec = 0 
frameRate = 0.1                               #  10 frames per second
success = getFrame(sec) 

while success:     
    print("*********************************************************************************")                           
    sec += frameRate 
    sec = round(sec, 2)   
    ret, frame = getFrame(sec) 
    seco = (str(sec)).replace('.', '_')
    cv2.imwrite('bike/20220531141406/frames/'+str(seco)+'.jpg',frame)   
    
    if not ret:
        break
   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
      
    date_time_obj = datetime.datetime.strptime(START_Videos_Time, '%Y-%m-%d %H:%M:%S')
    sec = round(sec, 2)
    Full_TIme = convert(sec)
    print("date_time",date_time_obj + datetime.timedelta(seconds=Full_TIme[0],minutes=Full_TIme[1],hours=Full_TIme[1]))
                
    l = len(sheet.col_values(1)) 
    l += 1
    tlist = []
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):  
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            vid = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
            cv2.imwrite('bike/20220531141406/box/'+str(seco)+'.jpg',vid)
            crop = vid[ymin:ymax, xmin:xmax]
            path = "bike/20220531141406/crop/"+str(seco)+'.jpg'
            cv2.imwrite(path,crop)  

            imageSource=open(path,'rb')
            print(imageSource)
            response=client.detect_text(Image={'Bytes': imageSource.read()})
            textDetections=response['TextDetections']
            print ('Detected text\n----------')
            nm = []
            for text in textDetections:
                txt = text['DetectedText']
                print ('Detected text: ' + txt)
                # t = bool(re.search(r'^[A-Z]',txt))
                r = bool(re.search(r'\d{4}$',txt))
                if 'J' in txt or r and 'Id' in text: 
                    if text['Type']=='LINE':
                        nm.append(txt)
                        print("]]]]]]]]]]]]]]]] ", nm)
            if len(nm)==1:
                tlist = nm
                print("car Number plate Recognised:  ", tlist)
            elif len(nm)>=1:
                n = str(nm[0])+str(nm[1])
                tlist.append(n)
                print('bike Number plate Recognised:  ', tlist)
            else:
                print("Not Reconised but Detected............!!!")
                pass
                    
    if len(tlist)>0:
        tlist.insert(0,str(date_time_obj + datetime.timedelta(seconds=Full_TIme[0],minutes=Full_TIme[1],hours=Full_TIme[1])))
        print("tlist :::::::::",tlist)
        sheet.update('A'+str(l),[tlist])
