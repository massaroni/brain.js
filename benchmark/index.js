//const brainjs = require('../src').default;
const brainjs = require('../dist/index').default;

const {
  readMNISTFiles,
  mnistToBrainjsTrainingSet
} = require('./mnist-utils');

const dbPath = './mnist-data';
console.log('Reading MNIST database from ' + dbPath);
const mnistTrainingSet = readMNISTFiles(dbPath);
if (!mnistTrainingSet || mnistTrainingSet.length < 1) {
  return console.error('Empty training set.');
}

let brainTrainingSet = mnistToBrainjsTrainingSet(mnistTrainingSet);
//brainTrainingSet = brainTrainingSet.slice(0, 10);
console.log('Loaded', brainTrainingSet.length, 'training items.');
console.log(brainTrainingSet[0].input.length, 'Input neurons.');

const config = {
  binaryThresh: 0.5,
  hiddenLayers: [900, 200],
  activation: 'tanh',
  //activation: 'sigmoid',
  errorThresh: 0.005,
  iterations: 1000, // this is passed down to the trainer threads in multithreaded mode
  learningRate: 0.15,
  momentum: 0.1,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  logPeriod: 1,
  log: (...args) => {console.log('progress: ', ...args)}
};

//trainSingleThreaded(config);
trainMultithreaded(config, 7);

function trainSingleThreaded(config) {
  let syncConfig = Object.assign({}, config);
  let itemIterations = 0;
  const statusRegex = /iterations: (\d+), training error: (\d+(\.\d+)?)/;
  syncConfig.log = (statusStr) => {
    console.log(statusStr);
    const status = statusRegex.exec(statusStr);
    const iterations = parseInt(status[1]);
    itemIterations += iterations * brainTrainingSet.length;
    console.log('itemIterations:', itemIterations);
  };

  console.log('Training in single threaded mode...');
  const startMs = Date.now();
  const net = new brainjs.NeuralNetwork(syncConfig);
  const results = net.train(brainTrainingSet, syncConfig);
  const endMs = Date.now();
  console.log('Final itemIterations:', itemIterations);
  console.log('Done in', (endMs - startMs) / 1000, 'seconds:', JSON.stringify(results));
}

function trainMultithreaded(config, threadCount) {
  let asyncConfig = Object.assign({}, config);
  asyncConfig.parallel = {threads: threadCount, epochs: 100};
  asyncConfig.logPeriod = 1;
  asyncConfig.log = console.log;

  const startMs = Date.now();
  const net = new brainjs.NeuralNetwork(asyncConfig);
  
  net.trainAsync(brainTrainingSet, asyncConfig).then((results) => {
    const endMs = Date.now();
    console.log('Final:', results);
    console.log('Done in', (endMs - startMs) / 1000, 'seconds');
  }, (reason) => {
    console.error('Multithreded training failed: ', JSON.stringify(reason))
  });
}
