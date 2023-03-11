from flask import Flask, render_template, request
import cv2
from PIL import Image
import filters
import frequency
import histograms

app = Flask(__name__)

filterOperationsDict = {
    "gaussianNoise":filters.gaussian_noise,
    "uniformNoise":filters.uniform_noise,
    "saltAndPepper": filters.salt_pepper_noise,
    "gaussianFilter":filters.gaussian_filter,
    "averageFilter":filters.average_filter,
    "medianFilter":filters.median_filter,
    "canny": filters.canny_edge_detection,
    "prewitt":filters.prewitt_edge_detection,
    "roberts":filters.roberts_edge_detection,
    "sobel":filters.sobel_edge_detection
}

histogramsOperationsDict = {
    "histogramPlot" : histograms.histogram,
    "equalize": histograms.equalization,
    "normalize":histograms.normalization,
    "localThreshold":histograms.local_threshold,
    "globalThreshold":histograms.global_threshold
}

def executeFilterOperation(operation,image):
    return filterOperationsDict[operation](image)

def executeHistogramOperation(operation,image):
    return histogramsOperationsDict[operation](image)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tab1",methods = ["POST","GET"])
def tab1():
    # Save the image to the assets
    image1 = request.files['image1']
    image1Show = Image.open(image1)
    image1Show.save(f"static/assets/image1.png")

    # Get the input image and the operation to be executed
    input_image = cv2.imread("static/assets/image1.png",0)
    operation1 = request.form['operation1']
    operation2 = request.form['operation2']
    operation3 = request.form['operation3']

    # Retrieve the output image after executing the operation
    noisy_image = executeFilterOperation(operation1, input_image)
    cv2.imwrite("static/assets/noisy_image.png",noisy_image)

    filtered_image = executeFilterOperation(operation2, noisy_image)
    cv2.imwrite("static/assets/filtered_image.png",filtered_image)

    edge_image = executeFilterOperation(operation3, input_image)
    cv2.imwrite("static/assets/edge_image.png",edge_image)

    return "tab1"

@app.route("/tab2", methods = ["POST","GET"])
def tab2():
    # Save the image to the assets
    image1 = request.files['image1']
    image1Show = Image.open(image1)
    image1Show.save(f"static/assets/image1.png")

    # Get the input image and the operation to be executed
    input_image = cv2.imread("static/assets/image1.png",-1)
    operation1 = request.form['operation1']

    # Retrieve the output image after executing the operation
    output_image = executeHistogramOperation(operation1, input_image)
    cv2.imwrite("static/assets/histogram_image.png",output_image)
    
    return "tab2"

@app.route("/tab3", methods = ["POST","GET"])
def tab3():

    # Save the image to the assets
    image1 = request.files['image1']
    image2 = request.files['image2']

    image1Show = Image.open(image1)
    image1Show.save(f"static/assets/image1.png")

    image2Show = Image.open(image2)
    image2Show.save(f"static/assets/image2.png")

    # Get the input image and the operation to be executed
    input_image1 = cv2.imread("static/assets/image1.png",-1)
    input_image2 = cv2.imread("static/assets/image2.png",-1)

    output_image = frequency.hybrid_image(input_image1, input_image2)

    # Retrieve the output image after executing the operation
    cv2.imwrite("static/assets/hybrid_image.png",output_image) 

    return "tab3"

if __name__ == "__main__":
    app.run(debug = True)