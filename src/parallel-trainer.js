const partition = require('./utilities/partition');
const workerFarm = require('worker-farm');
const workers    = workerFarm(require.resolve('./parallel-trainer-worker'));
const { avgNetsRnnJson } = require('./utilities/avg-nets-rnn');
const { subtractNetsRnnJson } = require('./utilities/subtract-nets-rnn');
const { multNetRnnJson } = require('./utilities/scalar-mult-rnn');

const netNameToType = {
  NeuralNetwork: 'NeuralNetwork',
  NeuralNetworkGPU: 'NeuralNetworkGPU',
  //RNNTimeStep: 'recurrent.RNNTimeStep',
  LSTMTimeStep: 'recurrent.LSTMTimeStep',
  //GRUTimeStep: 'recurrent.GRUTimeStep',
  //RNN: 'recurrent.RNN',
  LSTM: 'recurrent.LSTM',
  //GRU: 'recurrent.GRU',
}

const aggregators = {
  NeuralNetwork: aggregatorNN,
  NeuralNetworkGPU: aggregatorNN,
  LSTM: aggregatorRNN,
  LSTMTimeStep: aggregatorRNN,
}

function aggregatorNN(previousJson, trainOpts, ...trainedNets) {
  return trainedNets[0].avg(...trainedNets.slice(1)).toJSON();
}

/**
 * RNN merging: V(t) = Va(t) - B*V(t-1)
 * roughly based on this: https://arxiv.org/pdf/1708.05604.pdf
 * 
 * @param {*} previousJson 
 * @param  {...any} trainedNets 
 */
function aggregatorRNN(previousJson, trainOpts, ...trainedNets) {
  const parallel = trainOpts.parallel || {};
  const rnnMergeBetaOpt = parallel.rnnMergeBeta;
  const rnnMergeBeta = Number.isFinite(rnnMergeBetaOpt) ? rnnMergeBetaOpt : 0.000001;
  const jsons = trainedNets.map((n) => n.toJSON());
  const avg = avgNetsRnnJson(...jsons);
  const previousJsonRescaled = multNetRnnJson(rnnMergeBeta, previousJson);
  return subtractNetsRnnJson(avg, previousJsonRescaled);
}

/**
 * Multithreaded training, via simple parameter averaging.
 */
async function trainParallel(data, net, trainOpts = {}) {
  const startMs = Date.now();
  const log = (trainOpts.log === true ? console.log : trainOpts.log) || (() => {});
  const logPeriod = trainOpts.logPeriod || 1;
  const parallel = trainOpts.parallel || {};
  const threadLog = parallel.log === true ? console.log : parallel.log || false;
  const NetCtor = Object.getPrototypeOf(net).constructor;
  const netType = NetCtor.name;
  const aggregator = aggregators[netType];
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
  
  net.prepTraining(data, trainOpts);
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

    globalWeights = aggregator(globalWeights, trainOpts, ...trainedNets);

    error = maxError;
    if (minError <= errorThresh) {
      if (parallel.errorMode === 'test') {
        const testnet = new NetCtor();
        testnet.fromJSON(globalWeights);
        if (!!testnet.test) {
          const testResult = testnet.test(data);
          error = Math.max(error, testResult.error);
        //} else if (maxError > errorThresh) {
        //  error = maxError;
        } else {
          const testIterations = 1;
          const testOpts = Object.assign({}, trainOpts);
          testOpts.iterations = testIterations;
          const result = testnet.train(data, testOpts);
          error = result.error;
          globalWeights = testnet.toJSON();
          iterations += testIterations;
          itemIterations += (data.length * testIterations);
        }
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

function unpackTrainOpts(trainOptions, net, data) {
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
    const type = netNameToType[netName];
    if (!type) {
      throw new Error('Unsupported net: ' + netName);
    }

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

    types.push({type, threadCount, partitions});
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
  const brainjs = require('./index');
  const ctor = brainjs.get(netType);
  const trained = new ctor();
  trained.fromJSON(netJSON);
  const status = trained.train(trainingData, trainOpts);
  return {status, trained};
}

function runTrainingWorker(netType, netJSON, trainingData, trainOpts) {
  const brainjs = require('./index');
  return new Promise((resolve, reject) => {
    workers({netType, netJSON, trainingData, trainOpts}, (error, results) => {
      if (error) {
        return reject(error);
      }

      const ctor = brainjs.get(netType);
      const trained = new ctor();
      trained.fromJSON(results.trainedNetJSON);

      resolve({trained, status: results.status});
    });
  });
}

module.exports = { trainParallel, unpackTrainOpts };
