const brainjs = require('../src');
const mathGenerator = require('./math-db-generator');
const shuffle = require('./shuffle-deterministic');

const brainTrainingSet = mathGenerator(5000, 'sin(x/3) + (0.5 * cos(x/2))', 1, -120, 120, true);
shuffle(brainTrainingSet, (x) => JSON.stringify(x.input));
console.log('Loaded', brainTrainingSet.length, 'training items.');
console.log(brainTrainingSet[0].input.length, 'Input neurons.');

const configs = [
  // {
  //   name: 'Single-thread 0.0001 LR',
  //   config: {
  //     iterations: 10000,
  //     learningRate: 0.0001,
  //     hiddenLayers: [50, 50],
  //     errorThresh: 0.005,
  //     activation: 'tanh',
  //     momentum: 0.5,
  //     binaryThresh: 0.5,
  //     beta1: 0.9,
  //     beta2: 0.999,
  //     epsilon: 1e-8,
  //     logPeriod: 1,
  //     log: true
  //   }
  // },

  {
    name: '2 Threads 0.0001 LR Overlapping Partitions',
    config: {
      parallel: {
        threads: 2,
        iterationsPerThread: 3,
        partitionSize: 3200,
        log: false,
        logPeriod: 1
      },
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
    }
  },

  {
    name: '4 Threads 0.0001 LR Overlapping Partitions',
    config: {
      parallel: {
        threads: 4,
        iterationsPerThread: 5,
        partitionSize: 2000,
        log: false,
        logPeriod: 1
      },
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
    }
  },

  {
    name: 'Single-thread 0.001 LR',
    config: {
      iterations: 10000,
      learningRate: 0.001,
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
    }
  },

  {
    name: '2 Threads 0.001 LR Overlapping Partitions',
    config: {
      parallel: {
        threads: 2,
        iterationsPerThread: 3,
        partitionSize: 3200,
        log: false,
        logPeriod: 1
      },
      iterations: 10000,
      learningRate: 0.001,
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
    }
  },

  {
    name: 'Single-thread 0.01 LR',
    config: {
      iterations: 10000,
      learningRate: 0.01,
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
    }
  },

  {
    name: '2 Threads 0.01 LR Overlapping Partitions',
    config: {
      parallel: {
        threads: 2,
        iterationsPerThread: 3,
        partitionSize: 3200,
        log: false,
        logPeriod: 1
      },
      iterations: 10000,
      learningRate: 0.01,
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
    }
  },

];

let allResults = [];
let c = 0;
function processNext() {
  if (c < configs.length) {
    runTraining(configs[c++]).then(processNext, fail);
  } else {
    printResults();
    process.exit();
  }
}
processNext();

function runTraining(config) {
  const name = config.name;
  console.log('////// STARTING ' + name + ' //////');

  const parallel = config.config.parallel;
  const net = new brainjs.NeuralNetwork(config.config);
  let itemIterations = 0;
  let threadCount = 0;
  const startMs = Date.now();

  let done = function (status) {
    const endMs = Date.now();
    const durationS = Math.floor((endMs - startMs) / 1000);
    const error = status.error;
    const testError = net.test(brainTrainingSet).error;
    const itemIterationsPerThread = Math.ceil(itemIterations / threadCount);
    allResults.push({config, durationS, error, testError, threadCount, itemIterationsPerThread, itemIterations});
    console.log('////// DONE ' + name + ' //////');
    console.log(durationS, 'seconds');
    console.log(itemIterationsPerThread, 'item iterations per thread');
    console.log(testError, 'testError')
    console.log(error, 'error');
  }

  config.config.logPeriod = 1;
  if (!!parallel) {
    config.config.log = function (status) {
      itemIterations = status.itemIterations;
      threadCount = status.threadCount;
      console.log('[' + name + '] error =', status.error);
      console.log('[' + name + '] item iterations =', itemIterations);
    };
    return net.trainAsync(brainTrainingSet, config.config).then(done, fail);
  }

  threadCount = 1;
  const statusRegex = /iterations: (\d+), training error: (\d+(\.\d+)?)/;
  config.config.log = (statusStr) => {
    console.log('[' + name + '] ' + statusStr);
    const status = statusRegex.exec(statusStr);
    if (!!status) {
      const iterations = parseInt(status[1]);
      itemIterations += iterations * brainTrainingSet.length;
      console.log('[' + name + '] item iterations =', itemIterations);
    }
  };
  done(net.train(brainTrainingSet, config.config));
  return Promise.resolve();
}

function fail(reason) {
  const msg = reason instanceof Error ? reason.message : JSON.stringify(reason);
  console.error('Benchmark failed: ', msg);
  process.exit();
}

function printResults() {
  console.log('////// Benchmark Results //////')
  for (let r of allResults) {
    console.log(r.config.name);
    console.log('     runtime = ', r.durationS, 'seconds');
    console.log('     item iterations per thread = ', r.itemIterationsPerThread);
    console.log('     error = ', r.error);
    console.log('     test error = ', r.testError);
  }
}
