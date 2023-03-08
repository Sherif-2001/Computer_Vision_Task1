from flask import Flask, render_template, request
skills_app = Flask(__name__, static_url_path='')


@skills_app.route("/")
def home():
    return render_template("index.html")

@skills_app.route("/test",methods=['POST'])
def test():
    image1=request.files['image1'].read()
    image2=request.files['image2'].read()
    operation= request.form['operation']
    return "diaa"

