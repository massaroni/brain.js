/**
 * Utilities for parsing the MNIST hand written digit database, from here: http://yann.lecun.com/exdb/mnist/
 * thanks, Stackoverflow: https://stackoverflow.com/questions/25024179/reading-mnist-dataset-with-javascript-node-js
 */

const fs = require('fs-extra');

module.exports = {
  readMNISTFiles,
  mnistToBrainjsTrainingSet
};

function readMNISTFiles(dirname = __dirname) {
  var dataFileBuffer = fs.readFileSync(dirname + '/train-images-idx3-ubyte');
  var labelFileBuffer = fs.readFileSync(dirname + '/train-labels-idx1-ubyte');
  var pixelValues = [];

  for (var image = 0; image < 60000; image++) {
    var pixels = [];

    for (var y = 0; y <= 27; y++) {
      for (var x = 0; x <= 27; x++) {
        pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
      }
    }

    var imageData = {};
    imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

    pixelValues.push(imageData);
  }

  return pixelValues;
}

function mnistToBrainjsTrainingSet(mnistTrainingSet) {
  return mnistTrainingSet.map(function (img) {
    for (output in img) {
      return {input: img[output], output: output};
    }
    throw 'Output not found in ' + JSON.stringify(img);
  });
}
