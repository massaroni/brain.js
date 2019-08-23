import partition from './utilities/partition';
const workerFarm = require('worker-farm');
const workers    = workerFarm(require.resolve('./parallel-trainer-worker'));

/**
 * Ensemble training, via simple parameter averaging.
 */
export async function trainParallel(data, net, trainOpts = {}) {
  const startMs = Date.now();
  const log = (trainOpts.log === true ? console.log : trainOpts.log) || (() => {});
  const logPeriod = trainOpts.logPeriod || 1;
  const parallel = trainOpts.parallel || {};
  const threadLog = parallel.log === true ? console.log : parallel.log || false;
  const NetCtor = Object.getPrototypeOf(net).constructor;
  const maxEpochs = trainOpts.iterations || 1000;
  const errorThresh = trainOpts.errorThresh || NetCtor.trainDefaults.errorThresh;
  const threads = unpackTrainOpts(trainOpts, net, data);
  const threadCount = threads.length;

  let threadTrainOpts = Object.assign({}, trainOpts);
  delete threadTrainOpts.parallel;
  delete threadTrainOpts.callback;
  threadTrainOpts.iterations = parallel.iterationsPerThread || 10;
  threadTrainOpts.log = threadLog;
  threadTrainOpts.logPeriod = parallel.logPeriod || 1;
  threadTrainOpts.timeout = !threadTrainOpts.timeout || threadTrainOpts.timeout === Infinity ? Number.MAX_SAFE_INTEGER : threadTrainOpts.timeout;
  
  net.verifyIsInitialized(data);
  let globalWeights = net.toJSON();

  let error = 1;
  let epochs = 0;
  let iterations = 0;
  let itemIterations = 0;

  while (epochs < maxEpochs && error >= errorThresh) {
    let promises = [];

    for (let thread of threads) {
      if (parallel.synchronous === true) {
        let result = runTrainingSync(thread.type, globalWeights, thread.partition, threadTrainOpts);
        promises.push(Promise.resolve(result));
      } else {
        promises.push(runTrainingWorker(thread.type, globalWeights, thread.partition, threadTrainOpts));
      }
    }

    const results = await Promise.all(promises);
    let maxError, minError;
    let trainedNets = [];
    for (let t = 0; t < threadCount; t++) {
      const trained = results[t].trained;
      const status = results[t].status;
      maxError = t === 0 ? status.error : Math.max(maxError, status.error);
      minError = t === 0 ? status.error : Math.min(minError, status.error);
      trainedNets.push(trained);
      iterations += status.iterations;
      itemIterations += status.iterations * threads[t].partition.length;
    }

    globalWeights = trainedNets[0].avg(...trainedNets.slice(1)).toJSON();

    error = maxError;
    if (minError <= errorThresh) {
      if (parallel.errorMode === 'test') {
        const testnet = new NetCtor();
        testnet.fromJSON(globalWeights);
        const testResult = testnet.test(data);
        error = Math.max(error, testResult.error);
      } else {
        error = minError;
      }
    }
    
    epochs++;
    if (epochs % logPeriod === 0) {
      log({
        threadIterations: iterations,
        iterations: epochs,
        itemIterations,
        trainedNetJSON: globalWeights,
        error, threadCount});
    }
  }

  net.fromJSON(globalWeights);
  const endMs = Date.now();
  const elapsedMs = endMs - startMs;
  return {
    threadIterations: iterations,
    iterations: epochs,
    itemIterations,
    trainedNetJSON: globalWeights,
    error, threadCount, elapsedMs};
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
