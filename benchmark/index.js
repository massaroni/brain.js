//const brainjs = require('../src').default;
const brainjs = require('../dist/index').default;
const mathGenerator = require('./math-db-generator');
const shuffle = require('./shuffle-deterministic');

const brainTrainingSet = mathGenerator(5000, 'sin(x/3) + (0.5 * cos(x/2))', 3, -120, 120, true);
shuffle(brainTrainingSet, (x) => JSON.stringify(x.input));
console.log('Loaded', brainTrainingSet.length, 'training items.');
console.log(brainTrainingSet[0].input.length, 'Input neurons.');

const multithreadedConfig = {
  parallel: {threads: 3, epochs: 10000, log: false, logPeriod: 1, syncMode: false},
  iterations: 6, // this is passed down to the trainer threads in multithreaded mode
  learningRate: 0.0001,
  hiddenLayers: [50, 50],
  errorThresh: 0.005,
  activation: 'tanh',
  momentum: 0.5,
  binaryThresh: 0.5,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  logPeriod: 1,
  log: true
};

const singlethrededConfig = {
  iterations: 10000,
  learningRate: 0.0001,
  hiddenLayers: [50, 50],
  errorThresh: 0.005,
  activation: 'tanh',
  momentum: 0.5,
  binaryThresh: 0.5,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  logPeriod: 1,
  log: true
};

trainSingleThreaded(singlethrededConfig).then(() => trainMultithreaded(multithreadedConfig)).then(() => process.exit());

function trainSingleThreaded(config) {
  let itemIterations = 0;
  const statusRegex = /iterations: (\d+), training error: (\d+(\.\d+)?)/;
  config.log = (statusStr) => {
    console.log('Single-threaded progress:' + statusStr);
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
  console.log('Single-threaded results: ', JSON.stringify(results));
  console.log('Single-threaded item iterations:', itemIterations, 'per thread.');
  return Promise.resolve();
}

function trainMultithreaded(config) {
  let itemIterations = 0;
  config.log = function (status) {
    itemIterations = status.itemIterations;
    console.log('Multi-threaded progress: error =', status.error);
    console.log('Multi-threaded item iterations =', itemIterations);
  };

  const net = new brainjs.NeuralNetwork(config);
  
  console.log('////// Training in multi-threaded mode //////');
  const startMs = Date.now();
  return net.trainAsync(brainTrainingSet, config).then((results) => {
    const endMs = Date.now();
    console.log('Done in', Math.floor((endMs - startMs) / 1000), 'seconds.');
    delete results.globalWeights;
    console.log('Multi-threaded results: ', JSON.stringify(results));
    console.log('Multi-threaded item iterations:', Math.ceil(itemIterations / results.threadCount), 'per thread.');
  }, (reason) => {
    const msg = reason instanceof Error ? reason.message : JSON.stringify(reason);
    console.error('Multi-threaded training failed: ', msg);
  });
}
