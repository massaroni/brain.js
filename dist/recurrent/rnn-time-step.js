'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Matrix = require('./matrix');
var RandomMatrix = require('./matrix/random-matrix');
var Equation = require('./matrix/equation');
var RNN = require('./rnn');

var RNNTimeStep = function (_RNN) {
  _inherits(RNNTimeStep, _RNN);

  // eslint-disable-next-line
  function RNNTimeStep(options) {
    _classCallCheck(this, RNNTimeStep);

    return _possibleConstructorReturn(this, (RNNTimeStep.__proto__ || Object.getPrototypeOf(RNNTimeStep)).call(this, options));
  }

  _createClass(RNNTimeStep, [{
    key: 'createInputMatrix',
    value: function createInputMatrix() {
      this.model.input = new RandomMatrix(this.inputSize, 1, 0.08);
    }
  }, {
    key: 'createOutputMatrix',
    value: function createOutputMatrix() {
      var model = this.model,
          outputSize = this.outputSize;

      var lastHiddenSize = this.hiddenSizes[this.hiddenSizes.length - 1];

      // whd
      model.outputConnector = new RandomMatrix(outputSize, lastHiddenSize, 0.08);
      // bd
      model.output = new Matrix(outputSize, 1);
    }
  }, {
    key: 'bindEquation',
    value: function bindEquation() {
      var model = this.model,
          hiddenSizes = this.hiddenSizes;
      var hiddenLayers = model.hiddenLayers;

      var equation = new Equation();
      var outputs = [];
      var equationConnection = model.equationConnections.length > 0 ? model.equationConnections[model.equationConnections.length - 1] : this.initialLayerInputs;

      // 0 index
      var output = RNNTimeStep.getEquation(equation, equation.input(model.input), equationConnection[0], hiddenLayers[0]);
      outputs.push(output);
      // 1+ indices
      for (var i = 1, max = hiddenSizes.length; i < max; i++) {
        output = RNNTimeStep.getEquation(equation, output, equationConnection[i], hiddenLayers[i]);
        outputs.push(output);
      }

      model.equationConnections.push(outputs);
      equation.add(equation.multiply(model.outputConnector, output), model.output);
      model.equations.push(equation);
    }

    /**
     *
     * @param {Number[]} input
     * @returns {number}
     */

  }, {
    key: 'runInput',
    value: function runInput(input) {
      this.runs++;
      var model = this.model;

      var errorSum = 0;
      var equation = void 0;
      while (model.equations.length < input.length - 1) {
        this.bindEquation();
      }
      var outputs = [];

      if (this.inputSize === 1) {
        for (var inputIndex = 0, max = input.length - 1; inputIndex < max; inputIndex++) {
          // start and end tokens are zeros
          equation = model.equations[inputIndex];

          var current = input[inputIndex];
          var next = input[inputIndex + 1];
          var output = equation.runInput([current]);
          for (var i = 0; i < output.weights.length; i++) {
            var error = output.weights[i] - next;
            // set gradients into log probabilities
            errorSum += Math.abs(error);

            // write gradients into log probabilities
            output.deltas[i] = error;
            outputs.push(output.weights);
          }
        }
      } else {
        for (var _inputIndex = 0, _max = input.length - 1; _inputIndex < _max; _inputIndex++) {
          // start and end tokens are zeros
          equation = model.equations[_inputIndex];

          var _current = input[_inputIndex];
          var _next = input[_inputIndex + 1];
          var _output = equation.runInput(_current);
          for (var _i = 0; _i < _output.weights.length; _i++) {
            var _error = _output.weights[_i] - _next[_i];
            // set gradients into log probabilities
            errorSum += Math.abs(_error);

            // write gradients into log probabilities
            _output.deltas[_i] = _error;
            outputs.push(_output.weights);
          }
        }
      }
      // this.model.equations.length - 1;
      this.totalCost = errorSum;
      return errorSum;
    }
  }, {
    key: 'runBackpropagate',
    value: function runBackpropagate() {
      for (var i = this.model.equations.length - 1; i > -1; i--) {
        this.model.equations[i].runBackpropagate();
      }
    }

    /**
     *
     * @param {Number[]|Number} [input]
     * @param {Number} [maxPredictionLength]
     * @param {Boolean} [isSampleI]
     * @param {Number} temperature
     * @returns {Number[]|Number}
     */

  }, {
    key: 'run',
    value: function run() /* , isSampleI = false, temperature = 1 */
    {
      var input = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
      var maxPredictionLength = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

      if (!this.isRunnable) return null;
      var model = this.model;

      while (model.equations.length < maxPredictionLength) {
        this.bindEquation();
      }
      var lastOutput = void 0;
      if (this.inputSize === 1) {
        for (var i = 0; i < input.length; i++) {
          var outputMatrix = model.equations[i].runInput([input[i]]);
          lastOutput = outputMatrix.weights;
        }
      } else {
        for (var _i2 = 0; _i2 < input.length; _i2++) {
          var _outputMatrix = model.equations[_i2].runInput(input[_i2]);
          lastOutput = _outputMatrix.weights;
        }
      }
      if (this.outputSize === 1) {
        return lastOutput[0];
      }
      return lastOutput;
    }

    /**
     *
     * @returns {Function}
     */

  }, {
    key: 'toFunction',
    value: function toFunction() {
      throw new Error(this.constructor.name + '-toFunction is not yet implemented');
    }
  }]);

  return RNNTimeStep;
}(RNN);

RNNTimeStep.defaults = {
  inputSize: 1,
  hiddenSizes: [20],
  outputSize: 1,
  learningRate: 0.01,
  decayRate: 0.999,
  smoothEps: 1e-8,
  regc: 0.000001,
  clipval: 5,
  json: null,
  dataFormatter: null
};

RNNTimeStep.trainDefaults = RNN.trainDefaults;

module.exports = RNNTimeStep;