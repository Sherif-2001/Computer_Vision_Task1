console.log("Hello there!");

const header = document.querySelector(".header");
const inputButtons = document.querySelectorAll(".input-btn");

const tab1Content = document.getElementById("tab1");
const tab2Content = document.getElementById("tab2");
const tab3Content = document.getElementById("tab3");

const tab1Button = document.getElementById("tab1-btn");
const tab2Button = document.getElementById("tab2-btn");
const tab3Button = document.getElementById("tab3-btn");

const imagesContainers = document.getElementById("images-containers");
const outputImageContainer = document.getElementById("output-img-container");
const edgeImageContainer = document.getElementById("edge-img-container");
const inputImagetwoContainer = document.getElementById("input-img-2-container");

const submitButton = document.getElementById("submit-btn");

const filterSelect = document.getElementById("filter-select");
const noiseSelect = document.getElementById("noise-select");
const edgeSelect = document.getElementById("edge-select");
const histogramSelect = document.getElementById("histogram-select");

const inputImageOne = document.getElementById("input-img-1");
const inputImageTwo = document.getElementById("input-img-2");
const outputImage = document.getElementById("output-img");
const edgeImage = document.getElementById("edge-img");

const files = [2];

inputButtons[0].addEventListener("change", function (_) {
  const file = inputButtons[0].files[0];
  files[0] = file;
  inputImageOne.classList.remove("hide");
  inputImageOne.src = URL.createObjectURL(file);
});

inputButtons[1].addEventListener("change", function (_) {
  const file = inputButtons[1].files[0];
  files[1] = file;
  inputImageTwo.classList.remove("hide");
  inputImageTwo.src = URL.createObjectURL(file);
});

tab1Button.addEventListener("click", function (_) {
  tab1Button.classList.add("active");
  tab2Button.classList.remove("active");
  tab3Button.classList.remove("active");
  tab1Content.classList.remove("hide");
  tab2Content.classList.add("hide");
  tab3Content.classList.add("hide");
  clearImages();
  header.textContent = "Frequency Filters";
  edgeImageContainer.classList.remove("hide")
});

tab2Button.addEventListener("click", function (_) {
  tab1Button.classList.remove("active");
  tab2Button.classList.add("active");
  tab3Button.classList.remove("active");
  tab1Content.classList.add("hide");
  tab2Content.classList.remove("hide");
  tab3Content.classList.add("hide");
  clearImages();
  header.textContent = "Histogram Operations";
  edgeImageContainer.classList.remove("hide");
});

tab3Button.addEventListener("click", function (_) {
  tab1Button.classList.remove("active");
  tab2Button.classList.remove("active");
  tab3Button.classList.add("active");
  tab1Content.classList.add("hide");
  tab2Content.classList.add("hide");
  tab3Content.classList.remove("hide");
  clearImages();
  header.textContent = "Hybrid images";
  edgeImageContainer.classList.add("hide");
});

function clearImages() {
  inputImageTwo.src = "";
  outputImage.src = "";
  edgeImage.src = "";
}

function submitClick(tabNum) {
  let formData = new FormData();
  formData.set("image1", files[0]);
  if (tabNum == 1) {
    formData.set("operation1", noiseSelect.value);
    formData.set("operation2", filterSelect.value);
    formData.set("operation3", edgeSelect.value);
  } else if (tabNum == 2) {
    formData.set("operation1", histogramSelect.value);
  } else if (tabNum == 3) {
    formData.set("image2", files[1]);
  }
  $.ajax({
    type: "POST",
    url: "/tab" + tabNum,
    enctype: "multipart/form-data",
    data: formData,
    processData: false,
    contentType: false,
    async: true,
    success: function (res) {
      if (tabNum == 1) {
        inputImageTwo.src = "static/assets/noisy_image.png?t=" + new Date().getTime();
        edgeImage.src = "static/assets/edge_image.png?t=" + new Date().getTime();
        outputImage.src = "static/assets/filtered_image.png?t=" + new Date().getTime();
      } else if (tabNum == 2) {
        inputImageTwo.src = "static/assets/histogram_plot.png?t=" + new Date().getTime();
        outputImage.src = "static/assets/histogram_image.png?t=" + new Date().getTime();
        edgeImage.src = "static/assets/edited_histogram_plot.png?t=" + new Date().getTime();
      }
      else if (tabNum == 3) {
        outputImage.src = "static/assets/hybrid_image.png?t=" + new Date().getTime();
      }
    },
  });
}

submitButton.addEventListener("click", function (_) {
  if (tab1Button.classList.contains("active")) submitClick(1);

  else if (tab2Button.classList.contains("active")) submitClick(2);

  else if (tab3Button.classList.contains("active")) submitClick(3);
});
