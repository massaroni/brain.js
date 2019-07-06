'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.trainParallel = undefined;

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

/**
 * Ensemble training, via simple parameter averaging.
 */
var trainParallel = exports.trainParallel = function () {
  var _ref = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee(data, net) {
    var trainOptions = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

    var startMs, log, logPeriod, parallel, NetCtor, maxEpochs, errorThresh, threads, peerTrainOptions, globalWeights, error, epochs, iterations, _trainedNets$, promises, _iteratorNormalCompletion, _didIteratorError, _iteratorError, _iterator, _step, thread, _result, results, worstError, trainedNets, threadCount, t, trained, status, partitionIdx, result, endMs, elapsedMs;

    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            startMs = Date.now();

            log = (trainOptions.log === true ? console.log : trainOptions.log) || function () {};

            logPeriod = trainOptions.logPeriod || 1;
            parallel = trainOptions.parallel || {};
            NetCtor = Object.getPrototypeOf(net).constructor;
            maxEpochs = parallel.epochs || 1000;
            errorThresh = trainOptions.errorThresh || NetCtor.trainDefaults.errorThresh;
            threads = unpackTrainOpts(trainOptions, net, data);
            peerTrainOptions = Object.assign({}, trainOptions);

            delete peerTrainOptions.parallel;
            delete peerTrainOptions.callback;
            delete peerTrainOptions.log;

            net.train([data[0]], { errorThresh: 0.9, iterations: 1 }); // initialize weights
            globalWeights = net.toJSON();
            error = 1;
            epochs = 0;
            iterations = 0;

          case 17:
            if (!(epochs < maxEpochs && error >= errorThresh)) {
              _context.next = 51;
              break;
            }

            promises = [];
            _iteratorNormalCompletion = true;
            _didIteratorError = false;
            _iteratorError = undefined;
            _context.prev = 22;


            for (_iterator = threads[Symbol.iterator](); !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
              thread = _step.value;

              if (parallel.syncMode === true) {
                _result = runTrainingSync(thread.type, globalWeights, thread.partition, peerTrainOptions);

                promises.push(Promise.resolve(_result));
              } else {
                promises.push(runTrainingWorker(thread.type, globalWeights, thread.partition, peerTrainOptions));
              }
            }

            _context.next = 30;
            break;

          case 26:
            _context.prev = 26;
            _context.t0 = _context['catch'](22);
            _didIteratorError = true;
            _iteratorError = _context.t0;

          case 30:
            _context.prev = 30;
            _context.prev = 31;

            if (!_iteratorNormalCompletion && _iterator.return) {
              _iterator.return();
            }

          case 33:
            _context.prev = 33;

            if (!_didIteratorError) {
              _context.next = 36;
              break;
            }

            throw _iteratorError;

          case 36:
            return _context.finish(33);

          case 37:
            return _context.finish(30);

          case 38:
            _context.next = 40;
            return Promise.all(promises);

          case 40:
            results = _context.sent;
            worstError = 0;
            trainedNets = [];
            threadCount = threads.length;

            for (t = threadCount - 1; t >= 0; t--) {
              trained = results[t].trained;
              status = results[t].status;

              trainedNets.push(trained);
              partitionIdx = (t === 0 ? threadCount : t) - 1;
              result = trained.test(threads[partitionIdx].partition[0]);

              worstError = Math.max(result.error, worstError);
              iterations += status.iterations;
            }
            error = worstError;
            epochs++;
            if (epochs % logPeriod === 0) {
              log('iterations: ' + iterations + ', error: ' + error + ', epochs: ' + epochs);
            }

            globalWeights = (_trainedNets$ = trainedNets[0]).avg.apply(_trainedNets$, _toConsumableArray(trainedNets.slice(1))).toJSON();
            _context.next = 17;
            break;

          case 51:

            net.fromJSON(globalWeights);
            endMs = Date.now();
            elapsedMs = endMs - startMs;
            return _context.abrupt('return', { error: error, iterations: iterations, epochs: epochs, elapsedMs: elapsedMs });

          case 55:
          case 'end':
            return _context.stop();
        }
      }
    }, _callee, this, [[22, 26, 30, 38], [31,, 33, 37]]);
  }));

  return function trainParallel(_x2, _x3) {
    return _ref.apply(this, arguments);
  };
}();

exports.unpackTrainOpts = unpackTrainOpts;

var _partition = require('./utilities/partition');

var _partition2 = _interopRequireDefault(_partition);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _asyncToGenerator(fn) { return function () { var gen = fn.apply(this, arguments); return new Promise(function (resolve, reject) { function step(key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { return Promise.resolve(value).then(function (value) { step("next", value); }, function (err) { step("throw", err); }); } } return step("next"); }); }; }

var workerFarm = require('worker-farm');
var workers = workerFarm(require.resolve('./parallel-trainer-worker'));function unpackTrainOpts(trainOptions, net, data) {
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