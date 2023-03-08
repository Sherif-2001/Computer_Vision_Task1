from flask import Flask, render_template, request
skills_app = Flask(__name__, static_url_path='')


@skills_app.route("/")
def home():
    return render_template("index.html")

@skills_app.route("/test",methods=['POST'])
def po():
    print('diaa')
    print(request.files['image1'])
    print(request.files['image2'])
    print(request.form['operation'])
    return "diaa"

