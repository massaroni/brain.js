const assert = require('assert');
const NeuralNetwork = require('../../src/neural-network');
const {trainParallel, unpackTrainOpts} = require('../../src/parallel-trainer');

describe('Parallel Trainer', () => {

  describe('training options', () => {
  
    it('can parse simple threads options', () => {
      const net = new NeuralNetwork();
      const trainingData = ['a','b','c','d','e','f'];
      const actual = unpackTrainOpts({parallel: {threads: 3}}, net, trainingData);

      const expectedThreads = [
        {type: 'NeuralNetwork', partition: ['a','b']},
        {type: 'NeuralNetwork', partition: ['c','d']},
        {type: 'NeuralNetwork', partition: ['e','f']},
      ];

      assert.deepEqual(expectedThreads, actual);
    });


    it('can generate overlapping partitions', () => {
      const net = new NeuralNetwork();
      const trainingData = ['a','b','c','d','e','f'];
      const actual = unpackTrainOpts({parallel: {threads: 3, partitionSize: 3}}, net, trainingData);

      const expectedThreads = [
        {type: 'NeuralNetwork', partition: ['a','b', 'c']},
        {type: 'NeuralNetwork', partition: ['b','c', 'd']},
        {type: 'NeuralNetwork', partition: ['d', 'e','f']},
      ];

      assert.deepEqual(expectedThreads, actual);
    });


    it('can parse complex threads options', () => {
      const net = new NeuralNetwork();
      const trainingData = ['a','b','c','d','e','f'];
      const actual = unpackTrainOpts({parallel:{
        threads: {
          NeuralNetwork: {
            threads: 2,
            trainingDataSize: 4,
            partitionSize: 3
          },
          NeuralNetworkGPU: 1
        }
      }}, net, trainingData);

      const expectedThreads = [
        {type: 'NeuralNetwork', partition: ['a','b','c']},
        {type: 'NeuralNetwork', partition: ['b','c','d']},
        {type: 'NeuralNetworkGPU', partition: ['e','f']},
      ];

      assert.deepEqual(expectedThreads, actual);
    });
    
  });


  describe('NeuralNetwork', () => {

    const checkPerformance = (net, trainingData) => {
      for (let i = 0; i < trainingData.length; i++) {
        let test1 = net.test(trainingData[i]);
        if (test1.misclasses.length > 0 || test1.error >= 0.1) {
          console.warn(JSON.stringify(test1));
        }
        assert.equal(test1.total, 1);
        assert.ok(test1.error < 0.1);
        assert.equal(test1.misclasses.length, 0);
      }
    };

    it('can converge via ensemble training.', async function () {
      const trainingData = [
        {input: [0, 1], output: [1]},
        {input: [0, 0], output: [0]},
        {input: [1, 0], output: [1]},
        {input: [1, 1], output: [0]}
      ];
  
      const net = new NeuralNetwork();
      const status = await trainParallel(trainingData, net, {parallel: { threads: 2, syncMode: true, errorMode: 'test'}});
      assert.ok(status.epochs >= 1);
      checkPerformance(net, trainingData);
    });


    it('can converge via multithreaded training with trainAsync()', async function () {
      const trainingData = [
        {input: [0, 1], output: [1]},
        {input: [0, 0], output: [0]},
        {input: [1, 0], output: [1]},
        {input: [1, 1], output: [0]}
      ];
  
      const net = new NeuralNetwork();
      const status = await net.trainAsync(trainingData, {parallel: { threads: 2, syncMode: true, errorMode: 'test'}});
      assert.ok(status.epochs >= 1);
      checkPerformance(net, trainingData);
    });

  });
});
