"use strict";
let classes = [];
let divisions = [];
let studentAttendance = [];
let students = [];
// let studentsSheet = "1yuZhqahTyB38ESFpWjMWoHSth3cNe7GOiJ71roPvZpw";
// let outPutSheet = "1TCXQzywuDtMXmMhAR3Hn-fcKmJ0GSnxvgqqbs71KfFM";
let studentsSheet = "1_qsjZmdO7eLyS1FjEYPHyeKvQOzoGm5-tX3ScVj5-OM";
let outPutSheet = "1t8ItS6KGK_d0s2cty40sWo71VlvZmeSb38HZvvCh-Os";

// Client ID and API key from the Developer //Console
// var CLIENT_ID =
//   "348555915453-8aa43gspn0v9s3stmhgjnfn81hrccdlj.apps.googleusercontent.com";
// var API_KEY = "AIzaSyAwWoGTtIHlHeVIlQsQieNVhxFEWwlX42Y";
var CLIENT_ID =
  "814884583044-ps9hco0c530fr9hag2settoqbdhppo80.apps.googleusercontent.com";
var API_KEY = "AIzaSyAsdxu5bqgMKIvPmgXVE3pOSXK0mkN96ok";

// Array of API discovery doc URLs for APIs used by the quickstart
var DISCOVERY_DOCS = [
  "https://sheets.googleapis.com/$discovery/rest?version=v4",
];

var SCOPES = "https://www.googleapis.com/auth/spreadsheets";

let authorizeButton = document.getElementById("authorize_button");
let signoutButton = document.getElementById("signout_button");
let classOption = document.getElementById("classOption");
let divOption = document.getElementById("divOption");

/**
 *  On load, called to load the auth2 library and API client library.
 */
function handleClientLoad() {
  // this is just temporary.
  // authorizeButton.style.display = "none";
  // signoutButton.style.display = "block";
  // classOption.style.display = "block";
  // divOption.style.display = "block";
  // listClassDiv();
  // $(".main-container").removeClass("login");
  gapi.load("client:auth2", initClient);
}

function initClient() {
  gapi.client
    .init({
      apiKey: API_KEY,
      clientId: CLIENT_ID,
      discoveryDocs: DISCOVERY_DOCS,
      scope: SCOPES,
    })
    .then(
      function () {
        // Listen for sign-in state changes.
        gapi.auth2.getAuthInstance().isSignedIn.listen(updateSigninStatus);

        // Handle the initial sign-in state.
        updateSigninStatus(gapi.auth2.getAuthInstance().isSignedIn.get());
        authorizeButton.onclick = handleAuthClick;
        signoutButton.onclick = handleSignoutClick;
      },
      function (error) {
        appendPre(JSON.stringify(error, null, 2));
      }
    );
}
if ("serviceWorker" in navigator) {
  window.addEventListener("load", function () {
    navigator.serviceWorker
      .register("/serviceWorker.js")
      .then((res) => console.log("service worker registered"))
      .catch((err) => console.log("service worker not registered", err));
  });
}

function updateSigninStatus(isSignedIn) {
  if (isSignedIn) {
    authorizeButton.style.display = "none";
    signoutButton.style.display = "block";
    classOption.style.display = "block";
    divOption.style.display = "block";
    listClassDiv();
    $(".main-container").removeClass("login");
    // $(".auth-container").hide();
    // $(".main-container").hide();
    // $(".container").show();
    //listClassStudents();
  } else {
    authorizeButton.style.display = "block";
    signoutButton.style.display = "none";
  }
}

/**
 *  Sign in the user upon button click.
 */
function handleAuthClick(event) {
  gapi.auth2.getAuthInstance().signIn();
}

/**
 *  Sign out the user upon button click.
 */
function handleSignoutClick(event) {
  gapi.auth2.getAuthInstance().signOut();
  location.reload();
}

function appendPre(message) {
  var pre = document.getElementById("content");
  var textContent = document.createTextNode(message + "\n");
  pre.appendChild(textContent);
}

function listMajors() {
  gapi.client.sheets.spreadsheets.values
    .get({
      spreadsheetId: "1mFT-D2uZ3G3AncWH7gEWogg1RzlEqE2tuAXjKHGotn0",
      range: "Class Data!A2:F",
    })
    .then(
      function (response) {
        var range = response.result;
        //console.log(range);
        if (range.values.length > 0) {
          // appendPre('Name, ,gender,Class Level ,Major, activity:');
          for (i = 0; i < range.values.length; i++) {
            var row = range.values[i];
            // Print columns A and E, which correspond to indices 0 and 4.
            //  appendPre(`${row[0]}- ${row[1]}-${row[2]}- ${row[4]}-${row[5]}`);
          }
        } else {
          //appendPre('No data found.');
        }
      },
      function (response) {
        //appendPre('Error: ' + response.result.error.message);
      }
    );
}
function listClassDiv() {
  gapi.client.sheets.spreadsheets.values
    .get({
      spreadsheetId: studentsSheet,
      range: "Student Master!I2:J",
    })
    .then(
      function (response) {
        const numberOfClass = response.result;
        // //console.log(numberOfClass);
        if (numberOfClass.values.length > 0) {
          populateClassOption(".classOption", numberOfClass);
          populateDivOption(".divOption", numberOfClass);
          // //appendPre('name -');
          for (let i = 0; i < numberOfClass.values.length; i++) {
            var row = numberOfClass.values[i];

            // Print columns A and E, which correspond to indices 0 and 4.
            //    //appendPre(`${row[0]} `);
            classes.push(row[0]);
          }
          //console.log(classes, "classes");
        } else {
          // //appendPre('No data found.');
        }
      },
      function (response) {
        //  //appendPre('Error: ' + response.result.error.message);
      }
    );
}

function appendSheetData(students) {
  let range = "Sheet1!A1:E";
  const request = {
    spreadsheetId: outPutSheet,
    range: range,
    resource: {
      values: students,
      majorDimension: "ROWS",
    },
    valueInputOption: "USER_ENTERED",
  };

  let appendCall = gapi.client.sheets.spreadsheets.values.append(request);
  appendCall.then(
    function (response) {
      swal({
        title: "Good job!",
        text: `${response.result.updates.updatedRows} attendance record successfully submitted`,
        icon: "success",
        button: "OK",
      }).then((value) => {
        studentAttendance = [];
        $(".student-table").hide();
        setTimeout(location.reload(), 3000);
      });
      //console.log(response.result.updates.updatedRows);
    },
    function (reason) {
      swal("error: " + reason.result.error.message);
    }
  );
}

async function addStudentToCard(studentList) {
  $(".tinder--cards").html("");
  let i = 0;
  //console.log(studentList, "stdlist");
  if (studentList) {
    studentList.forEach(async (student, index) => {
      ////console.log(student,"card generate");
      let studentCardData = `
            <div id="${index}" class="tinder--card">
                <h3>${student[1]}</h3>
                <p>${student[2]}-${student[3]} </p>
                <h6>Roll No - ${student[0]}</h6>

            </div>
          `;

      await $(".tinder--cards").append(studentCardData);
    });
  } else {
    swal("no students found");
  }

  nope.addEventListener("click", nopeListener);
  love.addEventListener("click", loveListener);
}
async function fetchClassWiseData() {
  let selectedClass = $("#classOption").find(":selected").val();
  let selectedDiv = $("#divOption").find(":selected").val();
  if (selectedClass && selectedDiv) {
    //console.log("both selected");
    await listClassStudents(selectedClass, selectedDiv);
  }
}
async function listClassStudents(c, d) {
  gapi.client.sheets.spreadsheets.values
    .get({
      spreadsheetId: studentsSheet,
      // range: `${c}${d}!A2:14`,
      range: `${c}${d}!A2:F`,
    })
    .then(
      async function (response) {
        students = response.result;
        //console.log("students : ", students.values);
        await addStudentToCard(students.values);
        allCards = document.querySelectorAll(".tinder--card");
        //console.log(allCards, "cards");
        initCards();

        allCards.forEach(function (el) {
          var hammertime = new Hammer(el);

          hammertime.on("pan", function (event) {
            el.classList.add("moving");
          });

          hammertime.on("pan", function (event) {
            if (event.deltaX === 0) return;
            if (event.center.x === 0 && event.center.y === 0) return;

            tinderContainer.classList.toggle("tinder_love", event.deltaX > 0);
            tinderContainer.classList.toggle("tinder_nope", event.deltaX < 0);

            var xMulti = event.deltaX * 0.03;
            var yMulti = event.deltaY / 80;
            var rotate = xMulti * yMulti;

            event.target.style.transform =
              "translate(" +
              event.deltaX +
              "px, " +
              event.deltaY +
              "px) rotate(" +
              rotate +
              "deg)";
          });

          hammertime.on("panend", function (event) {
            el.classList.remove("moving");
            tinderContainer.classList.remove("tinder_love");
            tinderContainer.classList.remove("tinder_nope");

            var moveOutWidth = document.body.clientWidth;
            var keep =
              Math.abs(event.deltaX) < 80 || Math.abs(event.velocityX) < 0.5;

            event.target.classList.toggle("removed", !keep);

            if (keep) {
              event.target.style.transform = "";
            } else {
              let studentIndex = students.values[event.target.id];
              //swipe events
              if (event.deltaX < 0) {
                studentIndex[4] = moment().format("M/DD/YYYY HH:mm:ss");
                studentIndex[5] = "absent";
                //console.log("absent student : ", studentIndex);
                studentAttendance.push(studentIndex);
              }
              if (event.deltaX > 0) {
                studentIndex[4] = moment().format("M/DD/YYYY HH:mm:ss");
                studentIndex[5] = "present";
                //console.log("present student : ", studentIndex);
                studentAttendance.push(studentIndex);
              }
              //console.log("studentAttendance : ", studentAttendance);
              checkStudentAttendance();
              // if (studentAttendance.length == students.values.length) {
              //   //console.log("Attendance is done..");
              //   $(".tinder").hide();
              //   addAttendanceToTable();
              // }
              var endX = Math.max(
                Math.abs(event.velocityX) * moveOutWidth,
                moveOutWidth
              );
              var toX = event.deltaX > 0 ? endX : -endX;
              var endY = Math.abs(event.velocityY) * moveOutWidth;
              var toY = event.deltaY > 0 ? endY : -endY;
              var xMulti = event.deltaX * 0.03;
              var yMulti = event.deltaY / 80;
              var rotate = xMulti * yMulti;

              event.target.style.transform =
                "translate(" +
                toX +
                "px, " +
                (toY + event.deltaY) +
                "px) rotate(" +
                rotate +
                "deg)";
              initCards();
            }
          });
        });

        //   await appendSheetData(students.values);
        if (students.values.length > 0) {
          //appendPre('name - class- division');
          for (let i = 0; i < students.values.length; i++) {
            var row = students.values[i];
            // Print columns A and E, which correspond to indices 0 and 4.
            //appendPre(`${row[0]} - ${row[1]} - ${row[2]}`);
          }
        } else {
          //appendPre('No data found.');
        }
      },
      function (response) {
        //appendPre('Error: ' + response.result.error.message);
      }
    );
}

function addAttendanceToTable() {
  //console.log("addAttendanceToTable : ");
  studentAttendance.forEach((student, index) => {
    let studentRow = `${
      student[5] == "present"
        ? `<tr id='tr${index}' class="present-student">`
        : `<tr id='tr${index}' class="absent-student">`
    }
                      <td style="font-weight: bold;">${student[0]}</td>
                      <td>${student[1]}</td>
                      <td>
                        <select class="form-select classOption" style="padding: 5px 35px 5px 5px;" onchange="updateAttendance(this)" id='${index}' class="attendance-select" aria-label="Default select example">
                          <option selected >${student[5]}</option>

                         ${
                           student[5] == "present"
                             ? '<option value="absent">absent</option>'
                             : '<option value="present">present</option>'
                         }
                         </select>
                      </td>
                    </tr>`;
    $(".students-row").append(studentRow);
  });
  $(".student-table").show();
  // $(".table-responsive").scroll(function () {
  //   var topPosition = $(".table-responsive").scrollTop();
  //   $(".table-responsive thead tr th").css({ top: topPosition });
  // });
}

function updateAttendance(e) {
  //console.log("tr element : ", e);
  $(`#tr${e.id}`).removeClass();
  $(`#tr${e.id}`).addClass(`${e.value}-student`);
  let updateAttendence = e.value || e.options[a.selectedIndex].value;
  let updateStudent = studentAttendance[e.id];
  updateStudent[5] = updateAttendence;
  //console.log("updated list of students : ", studentAttendance);
}

function populateClassOption(selector, Nubclass) {
  for (let i = 0; i < Nubclass.values.length; i++) {
    //console.log(Nubclass.values[i][0]);

    $(selector).append(
      `<option value="${Nubclass.values[i][0]}">${Nubclass.values[i][0]}</option>`
    );
  }
}

function populateDivOption(selector, Nubclass) {
  for (let i = 0; i < Nubclass.values.length; i++) {
    if (Nubclass.values[i][1]) {
      //console.log(Nubclass.values[i][1], "div");
      divisions.push(Nubclass.values[i][1]);
      $(selector).append(
        `<option value="${Nubclass.values[i][1]}">${Nubclass.values[i][1]}</option>`
      );
    }
  }
}

var tinderContainer = document.querySelector(".tinder");
var allCards = document.querySelectorAll(".tinder--card");

var nope = document.getElementById("nope");
var love = document.getElementById("love");

function initCards(card, index) {
  var newCards = document.querySelectorAll(".tinder--card:not(.removed)");

  newCards.forEach(function (card, index) {
    card.style.zIndex = allCards.length - index;
    card.style.transform =
      "scale(" + (20 - index) / 20 + ") translateY(-" + 30 * index + "px)";
    card.style.opacity = (10 - index) / 10;
  });

  tinderContainer.classList.add("loaded");
}

function createButtonListener(love) {
  return function (event) {
    var cards = document.querySelectorAll(".tinder--card:not(.removed)");
    var moveOutWidth = document.body.clientWidth * 1.5;

    if (!cards.length) return false;

    var card = cards[0];

    card.classList.add("removed");

    let studentIndex = students.values[card.id];
    //console.log("studentIndex with button : ", studentIndex);
    if (love) {
      //button event present
      studentIndex[4] = moment().format("M/DD/YYYY HH:mm:ss");
      studentIndex[5] = "present";
      //console.log("present student : ", studentIndex);
      studentAttendance.push(studentIndex);
      card.style.transform =
        "translate(" + moveOutWidth + "px, -100px) rotate(-30deg)";
    } else {
      studentIndex[4] = moment().format("M/DD/YYYY HH:mm:ss");
      studentIndex[5] = "absent";
      //console.log("absent student : ", studentIndex);
      studentAttendance.push(studentIndex);
      card.style.transform =
        "translate(-" + moveOutWidth + "px, -100px) rotate(30deg)";
    }

    checkStudentAttendance();

    initCards();

    event.preventDefault();
  };
}

function checkStudentAttendance() {
  if (studentAttendance.length == students.values.length) {
    //console.log("Attendance is done..");
    $(".tinder").hide();
    addAttendanceToTable();
  }
}

async function submitAttendance() {
  await appendSheetData(studentAttendance);

  //console.log("submit calls : ", studentAttendance);
}

var nopeListener = createButtonListener(false);
var loveListener = createButtonListener(true);

// nope.addEventListener("click", nopeListener);
// love.addEventListener("click", loveListener);

// let tf_lite_btn = document.getElementById("TfLitebtn");
// let objectDetector;
// tf_lite_btn.addEventListener("click", async function () {
//   const video = document.getElementById("videostream");
//   if (!objectDetector) {
//     const tfliteModel = await tflite.loadTFLiteModel("./assets/model.tflite");
//     // if (navigator.mediaDevices.getUserMedia) {
//     //   navigator.mediaDevices
//     //     .getUserMedia({ video: true })
//     //     .then((stream) => (video.srcObject = stream))
//     //     .catch((error) => console.log(error));
//     // }
//     const img = tf.browser.fromPixels(document.getElementById("tempImage"));
//     const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
//     let outputTensor = tfliteModel.predict(input);
//     console.log(outputTensor.dataSync());
//   }
//   console.log("objectDetector : ", objectDetector);
// });
let image_array = []

function saveFile(image) {
  image_array.push(image)
}


// function uploadFile(img)
function createFileObject(image) {
  
  return fetch(image)
  .then(res => res.blob())
  .then(blob => {
    return new File([blob], "image.png",{ type: "image/png" })
  })
}

function callAPI(imageFile){

  let formData = new FormData()
  console.log("formData in fn", formData);
  formData.append('file', imageFile)

  fetch("http://192.168.1.26:8012/input",
      { 
          body: formData,
          method: "post"
      });
}


async function uploadFile(img)  {
  
  let file_objects;
  // let formData = new FormData();

  file_objects = await createFileObject(img)

  console.log("in upload file function/..................", file_objects);

  console.log("callng api", file_objects)
  callAPI(file_objects)
  
}


let picture_click_btn = document.getElementById("click_image");
let camera_trigger = document.getElementById("camera_trigger");
let save_all_data = document.getElementById("saveAllData")

save_all_data.addEventListener("click", async function () {
  uploadFile(image_array)
})

var constraints = {
  video: {
    facingMode: { ideal: 'environment' },
    width: { ideal: 1280 },
    height: { ideal: 720 }
  },
  audio: false,
};
// var constraints = { video: { facingMode: "environment" }, audio: false };
const cameraView = document.querySelector("#camera--view");
const cameraOutput = document.querySelector("#camera--output");
const cameraSensor = document.querySelector("#camera--sensor");
var track = null;
// let objectDetector;
camera_trigger.addEventListener("click", async function () {
  console.log("camera trigger");
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(function (stream) {
        let stream_settings = stream.getVideoTracks()[0].getSettings();
        console.log("stream_settings.width:::::",stream_settings.width);
        console.log("stream_settings.height:::",stream_settings.height);
        // const para = document.createElement("h1");
        // para.innerHTML = stream_settings.width +" "+stream_settings.height;
        // Append to another element:
        // document.getElementById("myDIV").appendChild(para);

        track = stream.getTracks()[0];
        cameraView.srcObject = stream;
        console.log("videoWidth::",cameraView.videoWidth);
      })
      .catch(function (error) {
        console.log("Oops. Something is broken.", error);
      });
  }
});

$("#attendance-list-btn").on("click", function () {
  console.log("Submit button called : ", studentAttendance);
});

picture_click_btn.addEventListener("click", async function () {
  console.log("Clicked pictured : ", cameraView);
  $("#attendance-list-btn").css("display", "block");
  const tfliteModel = await tflite.ObjectDetector.create(
    "./assets/model.tflite"
  );
  let cameraSensor = document.getElementById("camera--sensor");
  cameraSensor.width = cameraView.videoWidth;
  cameraSensor.height = cameraView.videoHeight;
  // cameraSensor.width = 1080;
  // cameraSensor.height = 720;
  console.log("Sensor width: ",cameraSensor.width);
  console.log("Sensor height: ",cameraSensor.height);
  const para = document.createElement("h1");
  para.innerHTML = cameraSensor.width +" "+cameraSensor.height;
  document.getElementById("myDIV").appendChild(para);
  cameraSensor.getContext("2d").drawImage(cameraView, 0, 0);
  const ctx = cameraSensor.getContext("2d");

  // ctx.drawImage(cameraView, 0, 0, cameraView.width, cameraView.height);
  ctx.drawImage(cameraView, 0, 0, 1280, 720);
  let scannerQRCode = ctx.getImageData(
    0,
    0,
    // cameraView.videoWidth,
    // cameraView.videoHeight
    1080,
    720
  );
  let outputTensor = tfliteModel.detect(scannerQRCode, 0.3);
  // let newase64 = cameraSensor.toDataURL(scannerQRCode);
  // console.log("newase64 : ", newase64);
  // decodeQRCode(newase64);
  console.log("outputTensor::",outputTensor);
  let QRcoordinates = [];
  outputTensor.forEach((tensor) => {
    if (
      tensor.classes[0].probability >= 0.8 &&
      tensor.classes[0].probability <= 0.99
    ) {
      QRcoordinates.push(tensor.boundingBox);
    }
  });

  for (let i = 0; i < QRcoordinates.length; i++) {
    let coordinates = QRcoordinates[i];
    console.log("coordinates : ", coordinates);
    ctx.fillStyle = "green";
    ctx.beginPath();
    ctx.lineWidth = "8";
    ctx.strokeStyle = "green";
    ctx.rect(
      coordinates.originX,
      coordinates.originY,
      coordinates.width,
      coordinates.height
    );
    ctx.stroke();

    // let imageData = ctx.getImageData(
    //   coordinates.originX,
    //   coordinates.originY,
    //   coordinates.width,
    //   coordinates.height
    // );

    let cameraSensor1 = document.getElementById("camera--sensor");
    let canvas1 = document.createElement("canvas");
    let ctx1 = canvas1.getContext("2d");
    canvas1.width = coordinates.width;
    canvas1.height = coordinates.height;

    // var imageObj = new Image();
    // imageObj.onload = function() {
    //   ctx1.drawImage(cameraSensor1,
    //     coordinates.originX,
    //     coordinates.originY,
    //     coordinates.width,
    //     coordinates.height,
    //     0,
    //     0,
    //     coordinates.width,
    //     coordinates.height
    //     );
    //   };
      
      // ctx1.drawImage(canvas1, 0, 0, coordinates.width, coordinates.height);

    let imageData1 = ctx1.drawImage(cameraSensor1,
      coordinates.originX,
      coordinates.originY,
      coordinates.width,
      coordinates.height,
      0,
      0,
      coordinates.width,
      coordinates.height
      );
    console.log("imageData1:",imageData1);
    let context  =canvas1.getContext("2d")
    console.log("ctx1 : ",context );
    let newase64 = canvas1.toDataURL('image/png');
    // console.log("newase64 : ", newase64);
    // let base64Image = btoa(newase64);
    // decodeQRCode(newase64);
      // saveFile(newase64)
    uploadFile(newase64);
    //API call.....

   


  //   const rawResponse = await fetch('http://192.168.1.26:8012/input', {
  //   method: 'POST',
  //   headers: {
  //     'Accept': 'application/json',
  //     'Content-Type': 'application/json'
  //   },
  //   body: {'file': newase64}
  //   });
  //   const content = await rawResponse.json();
  //   console.log(content.Status);
  }
  // QRcoordinates.forEach((coordinates) => {
  //   console.log("coordinates : ", coordinates);
  //   ctx.fillStyle = "green";
  //   ctx.beginPath();
  //   ctx.lineWidth = "8";
  //   ctx.strokeStyle = "green";
  //   ctx.rect(
  //     coordinates.originX,
  //     coordinates.originY,
  //     coordinates.width,
  //     coordinates.height
  //   );
  //   ctx.stroke();
  //   let imageData = ctx.getImageData(
  //     coordinates.originX,
  //     coordinates.originY,
  //     coordinates.width,
  //     coordinates.height
  //   );
  //   let newase64 = cameraSensor.toDataURL(imageData);
  //   console.log("newase64 : ", newase64);
  //   decodeQRCode(newase64);
  // });
});

function decodeQRCode(scannerQRCode) {
  // console.log("scannerQRCode::",scannerQRCode);
  qrcode.decode(scannerQRCode);
  qrcode.callback = function (decodedDATA) {
    console.log("decodedDATA : ", decodedDATA);
    alert(decodedDATA);
  };
}
function onScanSuccess(decodedText, decodedResult) {
  // Handle on success condition with the decoded text or result.
  console.log(`Scan result: ${decodedText}`, decodedResult);
}