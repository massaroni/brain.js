'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

exports.trainParallel = trainParallel;
exports.unpackTrainOpts = unpackTrainOpts;

var _partition = require('./utilities/partition');

var _partition2 = _interopRequireDefault(_partition);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

var workerFarm = require('worker-farm');
var workers = workerFarm(require.resolve('./parallel-trainer-worker'));

/**
 * Ensemble training, via simple parameter averaging.
 */
async function trainParallel(data, net) {
  var trainOpts = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

  var startMs = Date.now();
  var log = (trainOpts.log === true ? console.log : trainOpts.log) || function () {};
  var logPeriod = trainOpts.logPeriod || 1;
  var parallel = trainOpts.parallel || {};
  var threadLog = parallel.log === true ? console.log : parallel.log || false;
  var NetCtor = Object.getPrototypeOf(net).constructor;
  var maxEpochs = parallel.epochs || 1000;
  var errorThresh = trainOpts.errorThresh || NetCtor.trainDefaults.errorThresh;
  var threads = unpackTrainOpts(trainOpts, net, data);
  var threadCount = threads.length;

  var threadTrainOpts = Object.assign({}, trainOpts);
  delete threadTrainOpts.parallel;
  delete threadTrainOpts.callback;
  threadTrainOpts.log = threadLog;
  threadTrainOpts.logPeriod = parallel.logPeriod || 1;
  threadTrainOpts.timeout = !threadTrainOpts.timeout || threadTrainOpts.timeout === Infinity ? Number.MAX_SAFE_INTEGER : threadTrainOpts.timeout;

  net.verifyIsInitialized(data);
  var globalWeights = net.toJSON();

  var error = 1;
  var epochs = 0;
  var iterations = 0;
  var itemIterations = 0;

  while (epochs < maxEpochs && error >= errorThresh) {
    var _trainedNets$;

    var promises = [];

    var _iteratorNormalCompletion = true;
    var _didIteratorError = false;
    var _iteratorError = undefined;

    try {
      for (var _iterator = threads[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
        var thread = _step.value;

        if (parallel.syncMode === true) {
          var result = runTrainingSync(thread.type, globalWeights, thread.partition, threadTrainOpts);
          promises.push(Promise.resolve(result));
        } else {
          promises.push(runTrainingWorker(thread.type, globalWeights, thread.partition, threadTrainOpts));
        }
      }
    } catch (err) {
      _didIteratorError = true;
      _iteratorError = err;
    } finally {
      try {
        if (!_iteratorNormalCompletion && _iterator.return) {
          _iterator.return();
        }
      } finally {
        if (_didIteratorError) {
          throw _iteratorError;
        }
      }
    }

    var results = await Promise.all(promises);
    var maxError = void 0,
        minError = void 0;
    var trainedNets = [];
    for (var t = 0; t < threadCount; t++) {
      var trained = results[t].trained;
      var status = results[t].status;
      maxError = t === 0 ? status.error : Math.max(maxError, status.error);
      minError = t === 0 ? status.error : Math.min(minError, status.error);
      trainedNets.push(trained);
      iterations += status.iterations;
      itemIterations += status.iterations * threads[t].partition.length;
    }

    globalWeights = (_trainedNets$ = trainedNets[0]).avg.apply(_trainedNets$, _toConsumableArray(trainedNets.slice(1))).toJSON();

    error = maxError;
    if (minError <= errorThresh) {
      if (parallel.errorMode === 'test') {
        var testnet = new NetCtor();
        testnet.fromJSON(globalWeights);
        var testResult = testnet.test(data);
        error = Math.max(error, testResult.error);
      } else {
        error = minError;
      }
    }

    epochs++;
    if (epochs % logPeriod === 0) {
      log({ iterations: iterations, error: error, epochs: epochs, itemIterations: itemIterations, threadCount: threadCount, globalWeights: globalWeights });
    }
  }

  net.fromJSON(globalWeights);
  var endMs = Date.now();
  var elapsedMs = endMs - startMs;
  return { error: error, iterations: iterations, itemIterations: itemIterations, epochs: epochs, threadCount: threadCount, elapsedMs: elapsedMs };
}

function unpackTrainOpts(trainOptions, net, data) {
  var parallel = trainOptions.parallel || {};
  var threadsOpts = parallel.threads;
  if (!threadsOpts || Number.isInteger(threadsOpts)) {
    var netCtorName = Object.getPrototypeOf(net).constructor.name;
    var threadsOptsObj = {};
    threadsOptsObj[netCtorName] = threadsOpts || 1;
    threadsOpts = threadsOptsObj;
  }

  var dataUsed = 0;
  var partitioned = 0;
  var types = [];
  var totalThreads = 0;
  for (var netName in threadsOpts) {
    var config = threadsOpts[netName];
    var threadCount = 1;
    var partitions = null;
    if ((typeof config === 'undefined' ? 'undefined' : _typeof(config)) === "object") {
      var partitionSize = config.partitionSize;
      threadCount = config.threads || 1;
      var trainingDataSize = Math.min(config.trainingDataSize || 0, data.length - dataUsed);

      if (trainingDataSize) {
        partitioned += threadCount;
        var trainingData = data.slice(dataUsed, trainingDataSize);
        partitions = (0, _partition2.default)(trainingData, threadCount, partitionSize);
        dataUsed += trainingDataSize;
      }
    } else if (Number.isInteger(config)) {
      threadCount = config || 1;
    }
    totalThreads += threadCount;

    types.push({ type: netName, threadCount: threadCount, partitions: partitions });
  }

  var unpartitioned = totalThreads - partitioned;
  if (unpartitioned) {
    var remainingData = dataUsed === 0 ? data : data.slice(dataUsed);
    var _partitionSize = parallel.partitionSize || 1;
    var _partitions = (0, _partition2.default)(remainingData, unpartitioned, _partitionSize);
    var _iteratorNormalCompletion2 = true;
    var _didIteratorError2 = false;
    var _iteratorError2 = undefined;

    try {
      for (var _iterator2 = types[Symbol.iterator](), _step2; !(_iteratorNormalCompletion2 = (_step2 = _iterator2.next()).done); _iteratorNormalCompletion2 = true) {
        var type = _step2.value;

        if (!type.partitions) {
          type.partitions = _partitions.splice(0, type.threadCount);
        }
      }
    } catch (err) {
      _didIteratorError2 = true;
      _iteratorError2 = err;
    } finally {
      try {
        if (!_iteratorNormalCompletion2 && _iterator2.return) {
          _iterator2.return();
        }
      } finally {
        if (_didIteratorError2) {
          throw _iteratorError2;
        }
      }
    }

    if (_partitions.length > 0) {
      throw new Error('Too many partitions');
    }
  }

  var threads = [];
  var _iteratorNormalCompletion3 = true;
  var _didIteratorError3 = false;
  var _iteratorError3 = undefined;

  try {
    for (var _iterator3 = types[Symbol.iterator](), _step3; !(_iteratorNormalCompletion3 = (_step3 = _iterator3.next()).done); _iteratorNormalCompletion3 = true) {
      var _type = _step3.value;

      var _partitions2 = _type.partitions;
      for (var t = 0; t < _type.threadCount; t++) {
        threads.push({ type: _type.type, partition: _partitions2.shift() });
      }
    }
  } catch (err) {
    _didIteratorError3 = true;
    _iteratorError3 = err;
  } finally {
    try {
      if (!_iteratorNormalCompletion3 && _iterator3.return) {
        _iterator3.return();
      }
    } finally {
      if (_didIteratorError3) {
        throw _iteratorError3;
      }
    }
  }

  return threads;
}

function runTrainingSync(netType, netJSON, trainingData, trainOpts) {
  var brainjs = require('./index').default;
  var ctor = brainjs[netType];
  var trained = new ctor();
  trained.fromJSON(netJSON);
  var status = trained.train(trainingData, trainOpts);
  return { status: status, trained: trained };
}

function runTrainingWorker(netType, netJSON, trainingData, trainOpts) {
  var brainjs = require('./index').default;
  return new Promise(function (resolve, reject) {
    workers({ netType: netType, netJSON: netJSON, trainingData: trainingData, trainOpts: trainOpts }, function (error, results) {
      if (error) {
        return reject(error);
      }

      var trained = new brainjs[netType]();
      trained.fromJSON(results.trainedNetJSON);

      resolve({ trained: trained, status: results.status });
    });
  });
}
//# sourceMappingURL=parallel-trainer.js.map