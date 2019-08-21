const {
  readMNISTFiles,
  mnistToBrainjsTrainingSet
} = require('./mnist-utils');

module.exports = function (dbPath = './mnist-data') {
  console.log('Reading MNIST database from ' + dbPath);
  
  const mnistTrainingSet = readMNISTFiles(dbPath);
  if (!mnistTrainingSet || mnistTrainingSet.length < 1) {
    return console.error('Empty training set.');
  }
  
  return mnistToBrainjsTrainingSet(mnistTrainingSet);
};
