var capture;
var video = document.getElementsByTagName("video")[0];
var stream;
{(async () => {
    stream = await navigator.mediaDevices.getUserMedia({video: true}); //prompts the user for permission to use a media input which produces a MediaStream with tracks containing the requested types of media
    video.srcObject = stream;
    video.play();
    capture = new ImageCapture(stream.getVideoTracks()[0]); //single media track within a stream
 })();}

 function send_capture(){
 var e = document.getElementById("cnn");
 var cnn_selected = e.options[e.selectedIndex].value;
 const socket = new WebSocket('ws://localhost:5678');
 socket.addEventListener('open', () => {
    const options = {imageWidth: 640, imageHeight: 480};
    capture.takePhoto(options).then(function(blob) {
    var reader = new FileReader();
    reader.readAsDataURL(blob); //blob to base64
    reader.onloadend = function() {
    var base64result = reader.result.split(',')[1]; //take only the part of base64 after coma
    console.time('measuredFunction');
    socket.send(
    JSON.stringify({
      net: cnn_selected, //to inform server which cnn will be used
      data: base64result}));
    }
  });


 socket.addEventListener('message', function (event) {
    console.timeEnd('measuredFunction');
    var result = document.getElementById("result");
    result.src = URL.createObjectURL(event.data);
    result.style.visibility = "visible"; //show result

});
 });}
 var camera = 1;
 $('.navbar-brand').on("click", function(){
        camera = camera +1;
        if(camera % 2 == 0){
         $(this).css("color", "red");
         video.pause();
         video.src = "";
         stream.getTracks()[0].stop();}
         else{
             $(this).css("color", "#dbdbdb");
             getmedia();
         }
    });

async function getmedia(){
    stream = await navigator.mediaDevices.getUserMedia({video: true});
    video.srcObject = stream;
    video.play();
    capture = new ImageCapture(stream.getVideoTracks()[0]);};

