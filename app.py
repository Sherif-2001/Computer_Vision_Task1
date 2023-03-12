import base64
import io
from flask import Flask, render_template, request, send_file
import cv2
from PIL import Image
import filters
import frequency
import histograms
import matplotlib.pyplot as plt


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
    "equalize": histograms.equalization,
    "normalize":histograms.normalization,
    "localThreshold":histograms.local_threshold,
    "globalThreshold":histograms.global_threshold,
    "RGBHistograms":histograms.rgb_hist_cumulative,
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
    input_image = cv2.imread("static/assets/image1.png",0)
    operation1 = request.form['operation1']

    # Retrieve the output image after executing the operation
    if operation1 == "RGBHistograms":
        output_image  = executeHistogramOperation(operation1, cv2.imread("static/assets/image1.png",-1))
    else:
        output_image  = executeHistogramOperation(operation1, input_image)

    cv2.imwrite("static/assets/histogram_image.png",output_image)
    
    histograms.saveHistogramPlot(input_image, 1)
    histograms.saveHistogramPlot(output_image, 2)

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

    # executing the frequency filters on the two images
    output_image = frequency.hybrid_image(input_image1, input_image2)

    # Retrieve the output image after executing the operation
    img_normalized = cv2.normalize(output_image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imsave("static/assets/hybrid_image.png",img_normalized )


    return "tab3"

if __name__ == "__main__":
    app.run(debug = True)