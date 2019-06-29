import partition from './utilities/partition';
import NeuralNetwork from './neural-network';
const workerFarm = require('worker-farm');
const workers    = workerFarm(require.resolve('./parallel-trainer-worker'));

/**
 * Ensemble training, via simple parameter averaging.
 */
export default async function trainParallel(data, net, trainOptions = {}) {
  const startMs = Date.now();
  const log = (trainOptions.log === true ? console.log : trainOptions.log) || (() => {});
  const logPeriod = trainOptions.logPeriod || 1;
  const pOptions = trainOptions.parallel || {};
  const threadCount = pOptions.threads || 1;
  const minPartitionSize = Math.ceil(data.length / threadCount);
  const partitionSize = Math.max(Math.ceil(pOptions.partitionSize || 1), minPartitionSize);
  const partitions = partition(data, threadCount, partitionSize);
  const NetCtor = Object.getPrototypeOf(net).constructor;
  const maxEpochs = pOptions.maxEpochs || 1000;
  const errorThresh = trainOptions.errorThresh || NetCtor.trainDefaults.errorThresh;
  
  let peerTrainOptions = Object.assign({}, trainOptions);
  delete peerTrainOptions.parallel;
  delete peerTrainOptions.callback;
  delete peerTrainOptions.log;
  
  net.train([data[0]], {errorThresh: 0.9, iterations: 1}); // initialize weights
  let globalWeights = net.toJSON();

  let error = 1;
  let epochs = 0;
  let iterations = 0;

  while (epochs < maxEpochs && error >= errorThresh) {
    let promises = [];

    for (let t = 0; t < threadCount; t++) {
      if (pOptions.syncMode === true) {
        let result = runTrainingSync(globalWeights, partitions[t], peerTrainOptions);
        promises.push(Promise.resolve(result));
      } else {
        promises.push(runTrainingWorker('NeuralNetwork', globalWeights, partitions[t], peerTrainOptions));
      }
    }

    const results = await Promise.all(promises);
    let worstError = 0;
    let trainedNets = [];
    for (let t = threadCount - 1; t >= 0; t--) {
      const trained = results[t].trained;
      const status = results[t].status;
      trainedNets.push(trained);
      const partitionIdx = (t === 0 ? threadCount : t) - 1;
      const result = trained.test(partitions[partitionIdx][0]);
      worstError = Math.max(result.error, worstError);
      iterations += status.iterations;
    }
    error = worstError;
    epochs++;
    if (epochs % logPeriod === 0) {
      log('iterations: ' + iterations + ', error: ' + error + ', epochs: ' + epochs);
    }

    globalWeights = trainedNets[0].avg(...trainedNets.slice(1)).toJSON();
  }

  net.fromJSON(globalWeights);
  const endMs = Date.now();
  const elapsedMs = endMs - startMs;
  return {error, iterations, epochs, elapsedMs};
}

function runTrainingSync(netJSON, trainingData, trainOpts) {
  const trained = new NeuralNetwork();
  trained.fromJSON(netJSON);
  const status = trained.train(trainingData, trainOpts);
  return {status, trained};
}

function runTrainingWorker(netType = 'NeuralNetwork', netJSON, trainingData, trainOpts) {
  return new Promise((resolve, reject) => {
    workers({netType, netJSON, trainingData, trainOpts}, (error, results) => {
      if (error) {
        return reject(error);
      }

      const trained = new NeuralNetwork();
      trained.fromJSON(results.trainedNetJSON);

      resolve({trained, status: results.status});
    });
  });
}
