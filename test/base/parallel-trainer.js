import assert from 'assert';
import NeuralNetwork from '../../src/neural-network';
import trainParallel from '../../src/parallel-trainer';

describe('Parallel Trainer', () => {
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
      const status = await trainParallel(trainingData, net, {parallel: { threads: 2, syncMode: true }});
      assert.ok(status.epochs > 1);
      checkPerformance(net, trainingData);
    });


    it('can converge via ensemble training with trainAsync()', async function () {
      const trainingData = [
        {input: [0, 1], output: [1]},
        {input: [0, 0], output: [0]},
        {input: [1, 0], output: [1]},
        {input: [1, 1], output: [0]}
      ];
  
      const net = new NeuralNetwork();
      const status = await net.trainAsync(trainingData, {parallel: { threads: 2, syncMode: true }});
      assert.ok(status.epochs > 1);
      checkPerformance(net, trainingData);
    });

  });
});
