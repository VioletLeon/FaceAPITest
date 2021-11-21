// import nodejs bindings to native tensorflow,
const tensorflow = require('@tensorflow/tfjs-node');

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
const canvas = require('canvas');
const faceapi = require('@vladmandic/face-api');

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement, additionally an implementation
// of ImageData is required, in case you want to use the MTCNN
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const imageSrc = 'biden1.jpg';
const imageSrc2 = 'biden2.jpg';

const compare = () => {
  const detect = async () => {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./weights');
    await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');

    // ageGenderNet
    // faceExpressionNet
    // faceLandmark68Net
    // faceLandmark68TinyNet
    // faceRecognitionNet
    // ssdMobilenetv1
    // tinyFaceDetector
    // tinyYolov2

    console.log('Detect started');

    // Load Image
    const img = await canvas.loadImage(imageSrc);
    console.log('Load finished');

    // Detect Face
    let detect1 = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    const faceMatcher = new faceapi.FaceMatcher(detect1);

    // Load second image
    const img2 = await canvas.loadImage(imageSrc2);
    console.log('Second image loaded');

    let comparison = await faceapi
      .detectSingleFace(img2)
      .withFaceLandmarks()
      .withFaceDescriptor();
    console.log('Second image processed');

    const match = faceMatcher.findBestMatch(comparison.descriptor);

    console.log(match);
    return detect1;
  };

  return detect();
};

compare();
