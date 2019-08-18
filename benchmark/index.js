//const brainjs = require('../src').default;
const brainjs = require('../dist/index').default;
const mathGenerator = require('./math-db-generator');

const brainTrainingSet = mathGenerator(2000, 'sin(x/3) + (0.5 * cos(x/2))', 2, -100, 100, true);
console.log('Loaded', brainTrainingSet.length, 'training items.');
console.log(brainTrainingSet[0].input.length, 'Input neurons.');

const multithreadedConfig = {
  parallel: {threads: 2, epochs: 10000, log: true, logPeriod: 1, syncMode: true},
  iterations: 100, // this is passed down to the trainer threads in multithreaded mode
  binaryThresh: 0.5,
  hiddenLayers: [50, 50],
  activation: 'tanh',
  errorThresh: 0.005,
  learningRate: 0.001,
  momentum: 0.5,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  logPeriod: 1,
  log: (...args) => {console.log('progress: ', ...args)}
};

const singlethrededConfig = {
  iterations: 10000,
  binaryThresh: 0.5,
  hiddenLayers: [50, 50],
  activation: 'tanh',
  errorThresh: 0.005,
  learningRate: 0.001,
  momentum: 0.5,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  logPeriod: 1,
  log: (...args) => {console.log('progress: ', ...args)}
};

trainSingleThreaded(singlethrededConfig).then(() => trainMultithreaded(multithreadedConfig));

function trainSingleThreaded(config) {
  let itemIterations = 0;
  const statusRegex = /iterations: (\d+), training error: (\d+(\.\d+)?)/;
  config.log = (statusStr) => {
    console.log('Single-threaded progress:', statusStr);
    const status = statusRegex.exec(statusStr);
    if (!status) {
      console.log('Single-threaded status =', statusStr);
    } else {
      const iterations = parseInt(status[1]);
      itemIterations += iterations * brainTrainingSet.length;
      console.log('Single-threaded item iterations =', itemIterations);
    }
  };

  console.log('////// Training in single-threaded mode //////');
  const startMs = Date.now();
  const net = new brainjs.NeuralNetwork(config);
  const results = net.train(brainTrainingSet, config);
  const endMs = Date.now();
  console.log('Done in', Math.floor((endMs - startMs) / 1000), 'seconds.');
  console.log('Single-threaded results: ', results);
  console.log('Single-threaded item iterations:', itemIterations, 'per thread.');
  return Promise.resolve();
}

function trainMultithreaded(config) {
  let itemIterations = 0;
  config.log = function (status) {
    itemIterations = status.itemIterations;
    console.log('Multi-threaded progress:', status);
    console.log('Multi-threaded item iterations =', itemIterations);
  };

  const net = new brainjs.NeuralNetwork(config);
  
  console.log('////// Training in multi-threaded mode //////');
  const startMs = Date.now();
  return net.trainAsync(brainTrainingSet, config).then((results) => {
    const endMs = Date.now();
    console.log('Done in', Math.floor((endMs - startMs) / 1000), 'seconds.');
    console.log('Multi-threaded results: ', results);
    console.log('Multi-threaded item iterations:', Math.ceil(itemIterations / results.threadCount), 'per thread.');
  }, (reason) => {
    console.error('Multi-threaded training failed: ', JSON.stringify(reason))
  });
}
