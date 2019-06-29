const brain = require('../dist/index').default;

const xorTrainingData = [
  {input: [0, 1], output: [1]},
  {input: [0, 0], output: [0]},
  {input: [1, 0], output: [1]},
  {input: [1, 1], output: [0]}
];

const net = new brain.NeuralNetwork();

net.trainAsync(xorTrainingData, {log: true, logPeriod: 3, parallel: {threads: 3}}).then(console.log);
