from flask import Flask, render_template, request
import cv2
from PIL import Image
import base64
import filters
import frequency
import histograms

app = Flask(__name__)

filterOperationsDict = {
    "gaussianNoise":filters.gaussian_noise,
    "uniformNoise":filters.unifrom_noise,
    "saltAndPepper": filters.sp_noise,
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
    "histogramEqualization": histograms.equalization,
    "histogramNormalization":histograms.normalization,
    "localThresholding":histograms.local_threshold,
    "globalThresholding":histograms.global_threshold
}

def executeFilterOperation(operation,image):
    return filterOperationsDict[operation](image)

def executeHistogramOperation(operation,image):
    return histogramsOperationsDict[operation]()

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
    input_image = cv2.imread("static/assets/image1.png",-1)
    operation = request.form['operation']

    # Retrieve the output image after executing the operation
    output_image = executeFilterOperation(operation, input_image)
    cv2.imwrite("static/assets/filter_output_image.png",output_image)

    encoded_string = base64.b64encode(output_image)
    return encoded_string

@app.route("/tab2")
def tab2():
    # Save the image to the assets
    image1 = request.files['image1']
    image1Show = Image.open(image1)
    image1Show.save(f"static/assets/image1.png")

    # Get the input image and the operation to be executed
    input_image = cv2.imread("static/assets/image1.png",-1)
    operation = request.form['operation']

    # Retrieve the output image after executing the operation
    output_image = executeFilterOperation(operation, input_image)
    cv2.imwrite("static/assets/histogram_output_image.png",output_image)
    
    return "tab2"

@app.route("/tab3")
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

    output_image = frequency.hybrid_image(input_image1,input_image2)

    # Retrieve the output image after executing the operation
    cv2.imwrite("static/assets/hybrid_output_image.png",output_image)    

    return "tab3"

if __name__ == "__main__":
    app.run(debug = True)