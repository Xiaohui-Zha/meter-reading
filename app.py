from flask import Flask
import function

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>hello world </h1>'

@app.route('/lights')
def lights():
    img = cv2.imread('../img/light_2.jpg')
    lights = detection(img)
    return lights

@app.route('/led_nums')
def led_nums():
    img = img = cv2.imread('../real/30.jpg')
    return pred_nums(img)

@app.route('/single_track')
def single_track():
    retrun trackSingle()
