import partition from './utilities/partition';
const workerFarm = require('worker-farm');
const workers    = workerFarm(require.resolve('./parallel-trainer-worker'));

/**
 * Ensemble training, via simple parameter averaging.
 */
export async function trainParallel(data, net, trainOptions = {}) {
  const startMs = Date.now();
  const log = (trainOptions.log === true ? console.log : trainOptions.log) || (() => {});
  const logPeriod = trainOptions.logPeriod || 1;
  const parallel = trainOptions.parallel || {};
  const NetCtor = Object.getPrototypeOf(net).constructor;
  const maxEpochs = parallel.epochs || 1000;
  const errorThresh = trainOptions.errorThresh || NetCtor.trainDefaults.errorThresh;
  const threads = unpackTrainOpts(trainOptions, net, data);
  
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

    for (let thread of threads) {
      if (parallel.syncMode === true) {
        let result = runTrainingSync(thread.type, globalWeights, thread.partition, peerTrainOptions);
        promises.push(Promise.resolve(result));
      } else {
        promises.push(runTrainingWorker(thread.type, globalWeights, thread.partition, peerTrainOptions));
      }
    }

    const results = await Promise.all(promises);
    let worstError = 0;
    let trainedNets = [];
    const threadCount = threads.length;
    for (let t = threadCount - 1; t >= 0; t--) {
      const trained = results[t].trained;
      const status = results[t].status;
      trainedNets.push(trained);
      const partitionIdx = (t === 0 ? threadCount : t) - 1;
      const result = trained.test(threads[partitionIdx].partition[0]);
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

export function unpackTrainOpts(trainOptions, net, data) {
  const parallel = trainOptions.parallel || {};
  let threadsOpts = parallel.threads;
  if (!threadsOpts || Number.isInteger(threadsOpts)) {
    const netCtorName = Object.getPrototypeOf(net).constructor.name;
    let threadsOptsObj = {};
    threadsOptsObj[netCtorName] = threadsOpts || 1;
    threadsOpts = threadsOptsObj;
  }

  let dataUsed = 0;
  let partitioned = 0;
  let types = [];
  let totalThreads = 0;
  for (let netName in threadsOpts) {
    const config = threadsOpts[netName];
    let threadCount = 1;
    let partitions = null;
    if (typeof config === "object") {
      let partitionSize = config.partitionSize;
      threadCount = config.threads || 1;
      const trainingDataSize = Math.min(config.trainingDataSize || 0, data.length - dataUsed);

      if (trainingDataSize) {
        partitioned += threadCount;
        let trainingData = data.slice(dataUsed, trainingDataSize);
        partitions = partition(trainingData, threadCount, partitionSize);
        dataUsed += trainingDataSize;
      }
    } else if (Number.isInteger(config)) {
      threadCount = config || 1;
    }
    totalThreads += threadCount;

    types.push({type: netName, threadCount, partitions});
  }

  const unpartitioned = totalThreads - partitioned;
  if (unpartitioned) {
    const remainingData = dataUsed === 0 ? data : data.slice(dataUsed);
    const partitionSize = parallel.partitionSize || 1;
    const partitions = partition(remainingData, unpartitioned, partitionSize);
    for (let type of types) {
      if (!type.partitions) {
        type.partitions = partitions.splice(0, type.threadCount);
      }
    }
    if (partitions.length > 0) {
      throw new Error('Too many partitions');
    }
  }

  let threads = [];
  for (let type of types) {
    let partitions = type.partitions;
    for (let t = 0; t < type.threadCount; t++) {
      threads.push({type: type.type, partition: partitions.shift()});
    }
  }

  return threads;
}

function runTrainingSync(netType, netJSON, trainingData, trainOpts) {
  const brainjs = require('./index').default;
  const ctor = brainjs[netType];
  const trained = new ctor();
  trained.fromJSON(netJSON);
  const status = trained.train(trainingData, trainOpts);
  return {status, trained};
}

function runTrainingWorker(netType, netJSON, trainingData, trainOpts) {
  const brainjs = require('./index').default;
  return new Promise((resolve, reject) => {
    workers({netType, netJSON, trainingData, trainOpts}, (error, results) => {
      if (error) {
        return reject(error);
      }

      const trained = new brainjs[netType]();
      trained.fromJSON(results.trainedNetJSON);

      resolve({trained, status: results.status});
    });
  });
}
