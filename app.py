from flask import Flask, render_template, request
import cv2
from PIL import Image

app = Flask(__name__)

def saveImage(image, index):
    imageShow = Image.open(image)
    imageShow.save(f"static/assets/image{index}.png")

@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/test",methods=['POST'])
# def test():

#     image1 = request.files['image1']
#     saveImage(image1, 1)

#     image2 = request.files['image2']
#     saveImage(image2, 2)

#     operation = request.form['operation']
#     return "diaa"

@app.route("/tab1")
def tab1():
    image1 = request.files['image1']
    saveImage(image1, 1)

    

    return "tab1"

@app.route("/tab2")
def tab2():
    image1 = request.files['image1']
    saveImage(image1, 1)
    
    return "tab2"

@app.route("/tab3")
def tab3():
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    saveImage(image1, 1)
    saveImage(image2, 2)

    return "tab3"

if __name__ == "__main__":
    app.run(debug = True)


# @app.route("/image/<id>", methods = ["GET", "POST"])
# def uploadImage(id):
#     if(request.method == "POST"):
#         file = request.data
#         img = ImageSave.open(io.BytesIO(file))
#         if (id == "1"):
#         elif(id == "2"):
#     return "image processed"


# $.ajax({
#   type: "POST",
#   url: "/process_qtc",
#   data: JSON.stringify(server_data),
#   contentType: "application/json",
#   dataType: 'json' 
# });

# def process_qt_calculation():
#   if request.method == "POST":
#     qtc_data = request.get_json()
#     print(qtc_data)
#  results = {'processed': 'true'}
#  return jsonify(results)

