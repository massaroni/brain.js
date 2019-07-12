'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var lookup = require('./lookup');
// import TrainStream from './train-stream'
var mse2d = require('./utilities/mse-2d');
var layerFromJSON = require('./utilities/layer-from-json');
var _praxis = require('./praxis');
var flattenLayers = require('./utilities/flatten-layers');

var FeedForward = function () {
  _createClass(FeedForward, [{
    key: '_setLogMethod',


    /**
     *
     * @param log
     * if a method is passed in method is used
     * if false passed in nothing is logged
     * @returns error
     */
    value: function _setLogMethod(log) {
      if (typeof log === 'function') {
        this.trainOpts.log = log;
      } else if (log) {
        // eslint-disable-next-line
        this.trainOpts.log = console.log;
      } else {
        this.trainOpts.log = false;
      }
    }

    /**
     *
     * @param opts
     *    Supports all `trainDefaults` properties
     *    also supports:
     *       learningRate: (number)
     */

  }, {
    key: '_updateTrainingOptions',
    value: function _updateTrainingOptions(opts) {
      var _this = this;

      Object.keys(this.constructor.trainDefaults).forEach(function (opt) {
        _this.trainOpts[opt] = opts.hasOwnProperty(opt) ? opts[opt] : _this.trainOpts[opt];
      });
      this.constructor._validateTrainingOptions(this.trainOpts);
      this._setLogMethod(opts.log || this.trainOpts.log);
    }
  }], [{
    key: '_validateTrainingOptions',


    /**
     *
     * @param options
     * @private
     */
    value: function _validateTrainingOptions(options) {
      var validations = {
        iterations: function iterations(val) {
          return typeof val === 'number' && val > 0;
        },
        errorThresh: function errorThresh(val) {
          return typeof val === 'number' && val > 0 && val < 1;
        },
        log: function log(val) {
          return typeof val === 'function' || typeof val === 'boolean';
        },
        logPeriod: function logPeriod(val) {
          return typeof val === 'number' && val > 0;
        },
        learningRate: function learningRate(val) {
          return typeof val === 'number' && val > 0 && val < 1;
        },
        callback: function callback(val) {
          return typeof val === 'function' || val === null;
        },
        callbackPeriod: function callbackPeriod(val) {
          return typeof val === 'number' && val > 0;
        },
        timeout: function timeout(val) {
          return typeof val === 'number' && val > 0;
        }
      };
      Object.keys(FeedForward.trainDefaults).forEach(function (key) {
        if (validations.hasOwnProperty(key) && !validations[key](options[key])) {
          throw new Error('[' + key + ', ' + options[key] + '] is out of normal training range, your network will probably not train.');
        }
      });
    }
  }, {
    key: 'trainDefaults',
    get: function get() {
      return {
        iterations: 20000,
        errorThresh: 0.005,
        log: false,
        logPeriod: 10,
        learningRate: 0.3,
        callback: null,
        callbackPeriod: 10,
        reinforce: false
      };
    }
  }, {
    key: 'defaults',
    get: function get() {
      return {
        learningRate: 0.3,
        binaryThresh: 0.5,
        hiddenLayers: null,
        inputLayer: null,
        outputLayer: null,
        praxis: function praxis(layer) {
          return _praxis.momentumRootMeanSquaredPropagation(layer);
        }
      };
    }
  }, {
    key: 'structure',
    get: function get() {
      return {
        layers: null,
        _inputLayer: null,
        _outputLayer: null
      };
    }

    /**
     *
     * @param {object} options
     * @constructor
     */

  }]);

  function FeedForward() {
    var options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

    _classCallCheck(this, FeedForward);

    this.inputLayer = null;
    this.hiddenLayers = null;
    this.outputLayer = null;
    this.errorCheckInterval = 100;
    Object.assign(this, this.constructor.defaults, options);
    this.trainOpts = {};
    this._updateTrainingOptions(Object.assign({}, this.constructor.trainDefaults, options));
    Object.assign(this, this.constructor.structure);
  }

  _createClass(FeedForward, [{
    key: '_connectLayers',
    value: function _connectLayers() {
      var layers = [];
      this._inputLayer = this.inputLayer();
      var hiddenLayers = this._connectHiddenLayers(this._inputLayer);
      this._outputLayer = this.outputLayer(hiddenLayers[hiddenLayers.length - 1], hiddenLayers.length);
      layers.push(this._inputLayer);
      layers.push.apply(layers, _toConsumableArray(hiddenLayers));
      layers.push(this._outputLayer);
      this.layers = flattenLayers(layers);
    }
  }, {
    key: '_connectHiddenLayers',
    value: function _connectHiddenLayers(previousLayer) {
      var hiddenLayers = [];
      for (var i = 0; i < this.hiddenLayers.length; i++) {
        var hiddenLayer = this.hiddenLayers[i](previousLayer, i);
        hiddenLayers.push(hiddenLayer);
        previousLayer = hiddenLayer;
      }
      return hiddenLayers;
    }
  }, {
    key: 'initialize',
    value: function initialize() {
      this._connectLayers();
      this.initializeLayers(this.layers);
    }
  }, {
    key: 'initializeLayers',
    value: function initializeLayers(layers) {
      for (var i = 0; i < layers.length; i++) {
        var layer = layers[i];
        layer.setupKernels();
        if (layer.hasOwnProperty('praxis') && layer.praxis === null) {
          layer.praxis = this.praxis(layer);
        }
      }
    }

    /**
     *
     * @param input
     * @returns {*}
     */

  }, {
    key: 'run',
    value: function run(input) {
      if (this.inputLookup) {
        input = lookup.toArray(this.inputLookup, input);
      }

      var output = this.runInput(input);

      if (this.outputLookup) {
        output = lookup.toHash(this.outputLookup, output);
      }
      return output;
    }
  }, {
    key: 'runInput',
    value: function runInput(input) {
      this.layers[0].predict(input);
      for (var i = 1; i < this.layers.length; i++) {
        this.layers[i].predict();
      }
      return this.layers[this.layers.length - 1].weights;
    }

    /**
     *
     * @param data
     * @param options
     * @returns {{error: number, iterations: number}}
     */

  }, {
    key: 'train',
    value: function train(data) {
      var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      var status = void 0;
      var endTime = void 0;
      var _prepTraining2 = this._prepTraining(data, options);

      data = _prepTraining2.data;
      status = _prepTraining2.status;
      endTime = _prepTraining2.endTime;


      while (this._trainingTick(data, status, endTime)) {}
      return status;
    }

    /**
     *
     * @param {object} data
     * @param {object} status { iterations: number, error: number }
     * @param endTime
     */

  }, {
    key: '_trainingTick',
    value: function _trainingTick(data, status, endTime) {
      if (status.iterations >= this.trainOpts.iterations || status.error <= this.trainOpts.errorThresh || Date.now() >= endTime) {
        return false;
      }

      status.iterations++;

      if (this.trainOpts.log && status.iterations % this.trainOpts.logPeriod === 0) {
        status.error = this._calculateTrainingError(data);
        this.trainOpts.log('iterations: ' + status.iterations + ', training error: ' + status.error);
      } else if (status.iterations % this.errorCheckInterval === 0) {
        status.error = this._calculateTrainingError(data);
      } else {
        this._trainPatterns(data);
      }

      if (this.trainOpts.callback && status.iterations % this.trainOpts.callbackPeriod === 0) {
        this.trainOpts.callback(Object.assign(status));
      }
      return true;
    }

    /**
     *
     * @param data
     * @param options
     * @protected
     * @return { data, status, endTime }
     */

  }, {
    key: '_prepTraining',
    value: function _prepTraining(data, options) {
      this._updateTrainingOptions(options);
      data = this._formatData(data);
      var endTime = Date.now() + this.trainOpts.timeout;

      var status = {
        error: 1,
        iterations: 0
      };

      this.initialize();

      return {
        data: data,
        status: status,
        endTime: endTime
      };
    }

    /**
     *
     * @param data
     * @returns {Number} error
     */

  }, {
    key: '_calculateTrainingError',
    value: function _calculateTrainingError(data) {
      var sum = 0;
      for (var i = 0; i < data.length; ++i) {
        sum += this._trainPattern(data[i].input, data[i].output, true);
      }
      return sum / data.length;
    }

    /**
     * @param data
     * @private
     */

  }, {
    key: '_trainPatterns',
    value: function _trainPatterns(data) {
      for (var i = 0; i < data.length; ++i) {
        this._trainPattern(data[i].input, data[i].output, false);
      }
    }

    /**
     *
     * @param input
     * @param target
     * @param {Boolean} logErrorRate
     */

  }, {
    key: '_trainPattern',
    value: function _trainPattern(input, target, logErrorRate) {
      // forward propagate
      this.runInput(input);

      // back propagate
      this._calculateDeltas(target);
      this._adjustWeights();

      if (logErrorRate) {
        return mse2d(this._outputLayer.errors.hasOwnProperty('toArray') ? this._outputLayer.errors.toArray() : this._outputLayer.errors);
      }
      return null;
    }
  }, {
    key: '_calculateDeltas',
    value: function _calculateDeltas(target) {
      for (var i = this.layers.length - 1; i > -1; i--) {
        this.layers[i].compare(target);
      }
    }

    /**
     *
     */

  }, {
    key: '_adjustWeights',
    value: function _adjustWeights() {
      for (var i = 0; i < this.layers.length; i++) {
        this.layers[i].learn(this.layers[i - 1], this.layers[i + 1], this.trainOpts.learningRate);
      }
    }

    /**
     *
     * @param data
     * @returns {*}
     */

  }, {
    key: '_formatData',
    value: function _formatData(data) {
      var _this2 = this;

      if (!Array.isArray(data)) {
        // turn stream datum into array
        var tmp = [];
        tmp.push(data);
        data = tmp;
      }

      // turn sparse hash input into arrays with 0s as filler
      var datum = data[0].input;
      if (!Array.isArray(datum) && !(datum instanceof Float32Array)) {
        if (!this.inputLookup) {
          this.inputLookup = lookup.buildLookup(data.map(function (value) {
            return value.input;
          }));
        }
        data = data.map(function (datumParam) {
          var array = lookup.toArray(_this2.inputLookup, datumParam.input);
          return Object.assign({}, datumParam, { input: array });
        }, this);
      }

      if (!Array.isArray(data[0].output)) {
        if (!this.outputLookup) {
          this.outputLookup = lookup.buildLookup(data.map(function (value) {
            return value.output;
          }));
        }
        data = data.map(function (datumParam) {
          var array = lookup.toArray(_this2.outputLookup, datumParam.output);
          return Object.assign({}, datumParam, { output: array });
        }, this);
      }
      return data;
    }

    /**
     *
     * @param data
     * @returns {
     *  {
     *    error: number,
     *    misclasses: Array
     *  }
     * }
     */

  }, {
    key: 'test',
    value: function test() {
      throw new Error(this.constructor.name + '-test is not yet implemented');
    }

    /**
     *
     */

  }, {
    key: 'toJSON',
    value: function toJSON() {
      var jsonLayers = [];
      for (var i = 0; i < this.layers.length; i++) {
        var layer = this.layers[i];
        var jsonLayer = layer.toJSON();
        if (layer.hasOwnProperty('inputLayer')) {
          jsonLayer.inputLayerIndex = this.layers.indexOf(layer.inputLayer);
        } else if (layer.hasOwnProperty('inputLayer1') && layer.hasOwnProperty('inputLayer2')) {
          jsonLayer.inputLayer1Index = this.layers.indexOf(layer.inputLayer1);
          jsonLayer.inputLayer2Index = this.layers.indexOf(layer.inputLayer2);
        }
        jsonLayers.push(jsonLayer);
      }
      return {
        layers: jsonLayers
      };
    }

    /**
     *
     * @param json
     * @param [getLayer]
     * @returns {FeedForward}
     */

  }, {
    key: 'toFunction',


    /**
     *
     * @returns {Function}
     */
    value: function toFunction() {
      throw new Error(this.constructor.name + '-toFunction is not yet implemented');
    }

    /**
     * This will create a TrainStream (WriteStream) for us to send the training data to.
     * @param opts training options
     * @returns {TrainStream|*}
     */

  }, {
    key: 'createTrainStream',
    value: function createTrainStream() {
      throw new Error(this.constructor.name + '-createTrainStream is not yet implemented');
    }
  }], [{
    key: 'fromJSON',
    value: function fromJSON(json, getLayer) {
      var jsonLayers = json.layers;
      var layers = [];
      var inputLayer = layerFromJSON(jsonLayers[0]) || getLayer(jsonLayers[0]);
      layers.push(inputLayer);

      for (var i = 1; i < jsonLayers.length; i++) {
        var jsonLayer = jsonLayers[i];
        if (jsonLayer.hasOwnProperty('inputLayerIndex')) {
          var inputLayer1 = layers[jsonLayer.inputLayerIndex];
          layers.push(layerFromJSON(jsonLayer, inputLayer1) || getLayer(jsonLayer, inputLayer1));
        } else {
          if (!jsonLayer.hasOwnProperty('inputLayer1Index')) throw new Error('inputLayer1Index not defined');
          if (!jsonLayer.hasOwnProperty('inputLayer2Index')) throw new Error('inputLayer2Index not defined');
          var _inputLayer = layers[jsonLayer.inputLayer1Index];
          var inputLayer2 = layers[jsonLayer.inputLayer2Index];

          if (_inputLayer === undefined) throw new Error('layer of index ' + jsonLayer.inputLayer1Index + ' not found');
          if (inputLayer2 === undefined) throw new Error('layer of index ' + jsonLayer.inputLayer2Index + ' not found');

          layers.push(layerFromJSON(jsonLayer, inputLayer) || getLayer(jsonLayer, _inputLayer, inputLayer2));
        }
      }

      var net = new FeedForward(json);
      net.layers = layers;
      return net;
    }
  }]);

  return FeedForward;
}();

module.exports = FeedForward;