from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
def index():
    return "hello world!!"

@app.route("/tab1")
def tab1():
    return "tab1"

@app.route("/tab2")
def tab2():
    return "tab2"

@app.route("/tab3")
def tab3():
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


# <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js">
# </script>

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

