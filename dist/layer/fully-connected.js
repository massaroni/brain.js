'use strict';

var _get = function get(object, property, receiver) { if (object === null) object = Function.prototype; var desc = Object.getOwnPropertyDescriptor(object, property); if (desc === undefined) { var parent = Object.getPrototypeOf(object); if (parent === null) { return undefined; } else { return get(parent, property, receiver); } } else if ("value" in desc) { return desc.value; } else { var getter = desc.get; if (getter === undefined) { return undefined; } return getter.call(receiver); } };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Filter = require('./types').Filter;
var makeKernel = require('../utilities/kernel').makeKernel;
var values = require('../utilities/values');
var randos2D = require('../utilities/randos-2d');
var randos3D = require('../utilities/randos-3d');
var zeros = require('../utilities/zeros');
var zeros2D = require('../utilities/zeros-2d');
var zeros3D = require('../utilities/zeros-3d');

function predict(inputs, filters, biases) {
  var output = 0;
  var i = 0;
  for (var y = 0; y < this.constants.inputHeight; y++) {
    for (var x = 0; x < this.constants.inputWidth; x++) {
      output += inputs[y][x] * filters[this.thread.x][i];
      i++;
    }
  }
  return output + biases[this.thread.x];
}

function predict3D(inputs, filters, biases) {
  var output = 0;
  var i = 0;
  for (var z = 0; z < this.constants.inputDepth; z++) {
    for (var y = 0; y < this.constants.inputHeight; y++) {
      for (var x = 0; x < this.constants.inputWidth; x++) {
        output += inputs[z][y][x] * filters[this.thread.x][i];
        i++;
      }
    }
  }
  return output + biases[this.thread.x];
}

function compareInputDeltas(inputDeltas, deltas, filters) {
  var sum = 0;
  var filterX = this.thread.x + this.thread.y * this.output.x;
  for (var filterY = 0; filterY < this.constants.filterCount; filterY++) {
    sum += filters[filterY][filterX] * deltas[0][filterY];
  }
  return sum + inputDeltas[this.thread.y][this.thread.x];
}

function compareInputDeltas3D(inputDeltas, deltas, filters) {
  var sum = 0;
  var filterX = this.thread.x + this.thread.y * this.output.x;
  for (var filterY = 0; filterY < this.constants.filterCount; filterY++) {
    sum += filters[filterY][filterX] * deltas[0][filterY];
  }
  return sum + inputDeltas[this.thread.z][this.thread.y][this.thread.x];
}

function compareBiases(biases, deltas) {
  return biases[this.thread.x] + deltas[this.thread.y][this.thread.x];
}

function compareFilterDeltas(filterDeltas, inputWeights, deltas) {
  return filterDeltas[this.thread.y][this.thread.x] + inputWeights[this.thread.y][this.thread.x] * deltas[this.constants.deltaY][this.constants.deltaX];
}

function compareFilterDeltas3D(filterDeltas, inputWeights, deltas) {
  var inputZ = Math.floor(this.thread.x / (this.constants.inputWidth * this.constants.inputHeight));
  var inputY = Math.floor((this.thread.x - inputZ * this.constants.inputWidth * this.constants.inputHeight) / this.constants.inputWidth);
  var inputX = this.thread.x - this.constants.inputWidth * (inputY + this.constants.inputHeight * inputZ);
  return filterDeltas[this.thread.y][this.thread.x] + inputWeights[inputZ][inputY][inputX] * deltas[0][this.thread.y];
}

var FullyConnected = function (_Filter) {
  _inherits(FullyConnected, _Filter);

  _createClass(FullyConnected, null, [{
    key: 'defaults',
    get: function get() {
      return {
        bias: 0.1
      };
    }
  }]);

  function FullyConnected(settings, inputLayer) {
    _classCallCheck(this, FullyConnected);

    var _this = _possibleConstructorReturn(this, (FullyConnected.__proto__ || Object.getPrototypeOf(FullyConnected)).call(this, settings));

    _this.inputLayer = inputLayer;
    _this.validate();
    _this.compareFilterDeltasKernel = null;
    _this.compareInputDeltasKernel = null;
    _this.compareBiasesKernel = null;

    var connectionCount = inputLayer.width * inputLayer.height * inputLayer.depth;

    _this.biases = values(_this.height, _this.bias);
    _this.biasDeltas = zeros(_this.height);

    _this.filters = randos2D(connectionCount, _this.height);
    _this.filterDeltas = zeros2D(connectionCount, _this.height);

    if (_this.depth > 1) {
      _this.weights = randos3D(_this.width, _this.height);
      _this.deltas = zeros3D(_this.width, _this.height);
    } else if (_this.height > 1) {
      _this.weights = randos2D(_this.width, _this.height);
      _this.deltas = zeros2D(_this.width, _this.height);
    }
    return _this;
  }

  _createClass(FullyConnected, [{
    key: 'validate',
    value: function validate() {
      _get(FullyConnected.prototype.__proto__ || Object.getPrototypeOf(FullyConnected.prototype), 'validate', this).call(this);
      if (this.depth > 1) throw new Error('depth not supported');
    }
  }, {
    key: 'setupKernels',
    value: function setupKernels() {
      var inputLayer = this.inputLayer;

      var connectionCount = inputLayer.width * inputLayer.height * inputLayer.depth;
      if (inputLayer.depth > 1) {
        this.predictKernel = makeKernel(predict3D, {
          output: [this.width, this.height],
          constants: {
            inputHeight: inputLayer.height,
            inputWidth: inputLayer.width,
            inputDepth: inputLayer.depth
          }
        });

        this.compareFilterDeltasKernel = makeKernel(compareFilterDeltas3D, {
          output: [connectionCount, this.height],
          constants: {
            inputWidth: inputLayer.width,
            inputHeight: inputLayer.height
          }
        });

        this.compareInputDeltasKernel = makeKernel(compareInputDeltas3D, {
          output: [inputLayer.width, inputLayer.height, inputLayer.depth],
          constants: {
            filterCount: this.height
          }
        });
      } else {
        this.predictKernel = makeKernel(predict, {
          output: [this.width, this.height],
          constants: {
            inputHeight: inputLayer.height,
            inputWidth: inputLayer.width
          }
        });

        this.compareFilterDeltasKernel = makeKernel(compareFilterDeltas, {
          output: [connectionCount, this.height],
          constants: {
            inputWidth: inputLayer.width
          }
        });

        this.compareInputDeltasKernel = makeKernel(compareInputDeltas, {
          output: [inputLayer.width, inputLayer.height],
          constants: {
            filterCount: this.height
          }
        });
      }

      this.compareBiasesKernel = makeKernel(compareBiases, {
        output: [this.width, this.height]
      });
    }
  }, {
    key: 'predict',
    value: function predict() {
      this.weights = this.predictKernel(this.inputLayer.weights, this.filters, this.biases);
    }
  }, {
    key: 'compare',
    value: function compare() {
      this.inputLayer.deltas = this.compareInputDeltasKernel(this.inputLayer.deltas, this.deltas, this.filters);

      // TODO: handle biasDeltas learn
      this.biasDeltas = this.compareBiasesKernel(this.biases, this.deltas);

      // TODO: handle filterDeltas learn
      this.filterDeltas = this.compareFilterDeltasKernel(this.filterDeltas, this.inputLayer.weights, this.deltas);
    }
  }]);

  return FullyConnected;
}(Filter);

module.exports = { FullyConnected: FullyConnected, predict: predict, predict3D: predict3D, compareInputDeltas: compareInputDeltas, compareInputDeltas3D: compareInputDeltas3D, compareBiases: compareBiases, compareFilterDeltas: compareFilterDeltas, compareFilterDeltas3D: compareFilterDeltas3D };