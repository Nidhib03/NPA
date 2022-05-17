import os
import cv2
import numpy as np
import importlib.util
from app import app
from flask import request, jsonify
from werkzeug.utils import secure_filename
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import keras_ocr
from datetime import datetime

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("flowing-radio-347411-f6b13db7738b.json", scopes)
file = gspread.authorize(credentials) 
sheet = file.open("Number plate Recognition").get_worksheet(15)
header = ["Date", "Number plate"]
sheet.insert_row(header)

pipeline = keras_ocr.pipeline.Pipeline()

@app.route('/input-video', methods=['POST'])
def upload_video():
    vid_file = request.files['file']
    
    print("fie>>>>>>>>>>>", vid_file)
    if request.method == "POST":
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        vid_file = request.files['file']
        if vid_file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if vid_file:
            print("file name =====================", vid_file.name)
            now_vid = datetime.now()
            dt_string_vid = now_vid.strftime("%d/%m/%Y %H:%M:%S")
            filename = secure_filename("vid.webm" + dt_string_vid)
            vid_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = jsonify({"filename": filename, "Status":200})
            print("resp!!!!!!!!!!!!!!!",resp)
            resp.status_code = 200
                
            threshold=0.5
            GRAPH_NAME = 'model.tflite'
            LABELMAP_NAME = 'label.txt'
            VIDEO_NAME = 'Vid_store/' + filename
            min_conf_threshold = float(threshold)
            mainPath = '/home/rao/Flask/NPA/NPA_vid/'

            # Import TensorFlow libraries
            # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
            # If using Coral Edge TPU, import the load_delegate library
            pkg = importlib.util.find_spec('tflite_runtime')
            if pkg:
                from tflite_runtime.interpreter import Interpreter
            else:
                from tensorflow.lite.python.interpreter import Interpreter
                
            # Get path to current working directory
            CWD_PATH = os.getcwd()
            print('CWD_PATH:',CWD_PATH)
            # Path to video file
            VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
            print('video path:::::', VIDEO_PATH)
            # Path to .tflite file, which contains the model that is used for object detection
            PATH_TO_CKPT = os.path.join(mainPath,GRAPH_NAME)
            print('model file dir:::::',PATH_TO_CKPT)
            # Path to label map file
            PATH_TO_LABELS = os.path.join(mainPath,LABELMAP_NAME)
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
            fps= video.get(cv2.CAP_PROP_FPS)
            print(fps)
            # fps = int(fps)
            imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            f = 1           
            while f <= 10 and True:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                l = len(sheet.col_values(1)) 
                l += 1
                sheet.update('A'+str(l),dt_string)
                tlist = [] 
                ret, frame = video.read()
            
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
                for i in range(len(scores)):
                    
                    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                        vid = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                        crop = vid[ymin:ymax, xmin:xmax]

                        temp_NP_List = []
                        try:
                            images = [keras_ocr.tools.read(img) for img in [crop]]
                            prediction_groups = pipeline.recognize(images)
                            print("TRY")
                            for i in prediction_groups:
                                for text, box in i:
                                    print("text :::::::::", text)
                                    temp_NP_List.append(text)
                                temp_NP_List.remove('ind')
                        except:
                            print("except")
                            pass
                        print("temp_NP_List :::::::::::::::::",temp_NP_List)
                        num = (''.join(temp_NP_List))
                        print("Number plate:::",num)
                        tlist.append(num)
                        print(tlist)
                        print("one plate detected")
                        sheet.update('B'+str(l),[tlist])
                f += 1
                if not ret:
                    break
            return resp
        return resp 
    return 'file' 

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8011)))