console.log("Hello there!");

const header = document.querySelector(".header");
const input_btns = document.querySelectorAll(".input-btn");
const tab1Content = document.getElementById("tab1");
const tab2Content = document.getElementById("tab2");
const tab3Content = document.getElementById("tab3");
const tab1_btn = document.getElementById("tab1-btn");
const tab2_btn = document.getElementById("tab2-btn");
const tab3_btn = document.getElementById("tab3-btn");
const inputImageOne = document.querySelector(".input_img");
const inputOutput = document.querySelector(".input-output");
const outputImage = document.getElementById("output_img");
const submit_btn = document.getElementById("submit-btn");
const files = [2];

const filter_select = document.getElementById("filter-select");
const noise_select = document.getElementById("noise-select");

input_btns[0].addEventListener("change", function (e) {
  const file = input_btns[0].files[0];
  files[0] = file;
  inputImageOne.classList.remove("hide");
  inputImageOne.src = URL.createObjectURL(file);
});

input_btns[1].addEventListener("change", function (e) {
  const file = input_btns[1].files[0];
  console.log(file);
  files[1] = file;
  const inputImageTwo = document.getElementById("input-img-2");
  console.log(inputImageTwo);
  inputImageTwo.classList.remove("hide");
  inputImageTwo.src = URL.createObjectURL(file);
});

tab1_btn.addEventListener("click", function (e) {
  tab1_btn.classList.add("active");
  tab2_btn.classList.remove("active");
  tab3_btn.classList.remove("active");
  tab1Content.classList.remove("hide");
  tab2Content.classList.add("hide");
  tab3Content.classList.add("hide");

  header.textContent = "Frequency Filters";
  document.getElementById("input_img-2").classList.add("hide");
});

tab2_btn.addEventListener("click", function (e) {
  tab1_btn.classList.remove("active");
  tab2_btn.classList.add("active");
  tab3_btn.classList.remove("active");
  tab1Content.classList.add("hide");
  tab2Content.classList.remove("hide");
  tab3Content.classList.add("hide");

  header.textContent = "Histogram Operations";
  document.getElementById("input_img-2").classList.add("hide");
});

tab3_btn.addEventListener("click", function (e) {
  tab1_btn.classList.remove("active");
  tab2_btn.classList.remove("active");
  tab3_btn.classList.add("active");
  tab1Content.classList.add("hide");
  tab2Content.classList.add("hide");
  tab3Content.classList.remove("hide");

  header.textContent = "Hybrid images";
  document.getElementById("input_img-2").classList.remove("hide")
});

// tab1_btn.addEventListener("click", function (e) {

// });
// const input = document.getElementById("image_input")
// const output = document.getElementById("image_output")
// let imagesArray = []

// image_input.addEventListener("change", function(e) {
//     const file = input.files
//     imagesArray.push(file[0])
//     displayImages()
// })

// function displayImages(image) {
//     output.innerHTML = `<div class="image">
//     <img src="${URL.createObjectURL(image)}" alt="image">
//     </div>`
// }

function submitClick(tabNum) {
  let formData = new FormData();
  if (tabNum == 1) {
    formData.set("image1", files[0]);
    formData.set("operation1", noise_select.value);
    formData.set("operation2", filter_select.value)
  } else if (tabNum == 2) {
    formData.set("image1", files[0]);
    formData.set("operation1", noise_select.value);
  } else if (tabNum == 3) {
    formData.set("image1", files[0]);
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
    success: function (_) {
      outputImage.innerHTML = "";
      image = document.createElement("img");
      image.style.width = "100%";
      image.src = "static/assets/filtered_image.png?t=" + new Date().getTime();
      outputImage.appendChild(image);
    },
  });
}

console.log(submit_btn);
submit_btn.addEventListener("click", function (e) {
  if (tab1_btn.classList.contains("active")) submitClick(1);

  else if (tab2_btn.classList.contains("active")) submitClick(2);

  else if (tab3_btn.classList.contains("active")) submitClick(3);
});
