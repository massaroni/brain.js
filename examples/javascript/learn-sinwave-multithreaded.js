const brain = require('../../src');
const shuffle = require('../../benchmark/shuffle-deterministic');
const math = require('math-expression-evaluator');
const arrayEq = require('../../src/utilities/eq-shallow');

//// Configurable ////
const curves = makeCurves(200, 10, 'sin(x/6.5)', 1, 1, 0);
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
    trainedTest();
  }, function (reason) {
    console.error('ERROR:', reason);
  });
} else {
  net.train(curves, options);
  trainedTest();
}

function trainedTest(isVerbose) {
  const suffixLen = 1;
  let errors = 0;
  let avgError = 0;

  for (let i = 0; i < curves.length; i++) {
    const curve = curves[i];
    const input = curve.slice(0, curve.length - suffixLen);
    const expected = curve.slice(curve.length - suffixLen);
    let output = net.run(input);
    if (!Array.isArray(output)) {
      output = [output];
    }

    for (let j = 0; j < suffixLen; j++) {
      const predictedValue = output[j];
      const expectedValue = expected[j];

      if (predictedValue !== expectedValue) {
        const error = Math.abs((expectedValue - predictedValue) / expectedValue);
        avgError = avgError + (error - avgError) / (errors + 1);
        errors++;

        if (isVerbose && errors < 10) {
          console.log('error: ' + Math.round(error * 1000) / 10 + '%');
          console.log('expected', expectedValue);
          console.log('but was ', predictedValue);
          console.log('\n');
        }
      }
    }
  }
  console.log('Mismatches: ' + (errors / curves.length) * 100 + '%');
  console.log('Avg Error: ' + avgError * 100 + '%');
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

function normalizeNumber(n) {
  //return parseFloat(Math.fround(n).toString().substr(0, 8));
  return Math.fround(n);
}

function makeCurve(expression = 'sin(x)', xMin, length, xStep = 1) {
  const curve = [];
  for (let i = 0; i < length; i++) {
    const x = xMin + (xStep * i);
    const exp = expression.replace(/x/g, x.toString());
    const y = math.eval(exp);
    const yNormal = normalizeNumber(y);
    curve.push(yNormal);
  }

  return curve;
}

function makeCurves(curveCount, curveLen, expression = 'sin(x)', xSpacing = 1, xStep = 1, xMin = 0) {
  const curves = [];

  for (let i = 0; i < curveCount; i++) {
    const x = xMin + (xSpacing * i);
    curves.push(makeCurve(expression, x, curveLen, xStep));
  }

  shuffle(curves);
  return curves;
}

