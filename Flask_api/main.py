import os
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import keras_ocr
from datetime import datetime

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("flowing-radio-347411-f6b13db7738b.json", scopes)
file = gspread.authorize(credentials) 
sheet = file.open("Number plate Recognition").get_worksheet(15)
header = ["Date", "Number plate"]
sheet.insert_row(header)

pipeline = keras_ocr.pipeline.Pipeline()

def allowed_file(filename):
    print("in allowed file''''''''''", filename)
    allowd = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return allowd

@app.route('/input', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print("fie>>>>>>>>>>>", file)
    
    if request.method == "GET":
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            resp = jsonify({"Successfully GET file": filename})
            resp.status_code = 200
            return resp
        
    elif request.method == "POST":
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if file:
            print("file name =====================", file.name)
            filename = secure_filename("image.png")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = jsonify({"filename": filename, "Status":200})
            print("resp!!!!!!!!!!!!!!!",resp)
            resp.status_code = 200
            
            # for one image
            listimg = os.listdir('PostmanIMG/')
            image = ''.join(map(str,listimg))
            
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            tlist=[]
            l = len(sheet.col_values(1)) 
            l += 1
            sheet.update('A'+str(l),dt_string)
            
            temp_NP_List = []    
            try:
                images = [
                    keras_ocr.tools.read(img) for img in ['PostmanIMG/'+image]
                ]
                prediction_groups = pipeline.recognize(images)

                for i in prediction_groups:
                    for text, box in i:
                        print("text :::::::::", text)
                        temp_NP_List.append(text)
                    temp_NP_List.remove('ind')
            except:
                pass
            print("temp_NP_List :::::::::::::::::",temp_NP_List)
            num = (''.join(temp_NP_List))
            print("Number plate:::",num)
            tlist.append(num)
            print(tlist)
            print("one plate detected")
            sheet.update('B'+str(l),[tlist])
            
            return resp   
    return 'file' 
if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8012)))