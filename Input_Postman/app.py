from flask import Flask
from flask_cors import CORS


UPLOAD_FOLDER = 'PostmanIMG/'

app = Flask(__name__)
CORS(app)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024