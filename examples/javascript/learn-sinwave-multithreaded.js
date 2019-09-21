const brain = require('../../src');
const { makeSerieses, testTimeStep } = require('../../benchmark/timestep-rnn-utils');

//// Configurable ////
const curves = makeSerieses(200, 10, 'sin(x/6.5)', 1, 1, 0);
const multithreaded = true;
const options = {
  parallel: {
    threads: 7,
    iterationsPerThread: 4,
    //rnnMergeBeta: 0.0000008,
    errorMode: 'test',
    partitionSize: 50,
    log: false,
    logPeriod: 1
  },
  iterations: 200,
  hiddenLayers: [150],
  learningRate: 0.00005,
  errorThresh: 0.0002,
  log: logAsync,
  logPeriod: 1,
};
//// End Configurable ////

const LSTMTimeStep = brain.recurrent.LSTMTimeStep;
const net = new LSTMTimeStep(options);
let trainedJson;

if (multithreaded) {
  net.trainAsync(curves, options).then(function (status) {
    console.log('DONE');
    logAsync(status);
    testTimeStep(net, curves, 10);
  }, function (reason) {
    console.error('ERROR:', reason);
  });
} else {
  net.train(curves, options);
  testTimeStep(net, curves, 10);
}

function logAsync(status) {
  trainedJson = status.trainedNetJSON || trainedJson || net.toJSON();
  if (!this.printedNet) {
    console.log(JSON.stringify(trainedJson.options));
    this.printedNet = true;
  }
  delete status.trainedNetJSON;
  console.log(status);
}
