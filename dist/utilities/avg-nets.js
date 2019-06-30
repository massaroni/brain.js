'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = avgNets;

var _neuralNetwork = require('../neural-network');

var _neuralNetwork2 = _interopRequireDefault(_neuralNetwork);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Parameter averaging, supports NeuralNetwork and NeuralNetworkGPU.
 * @param  {...any} nets 
 */
function avgNets() {
  for (var _len = arguments.length, nets = Array(_len), _key = 0; _key < _len; _key++) {
    nets[_key] = arguments[_key];
  }

  if (!nets || !nets.length) {
    return;
  }

  if (nets.length === 1) {
    return nets[0];
  }

  var jsons = nets.map(function (net) {
    return net.toJSON();
  });

  if (!equalsShallow(jsons.map(function (json) {
    return json.sizes;
  }))) {
    throw new Error('Can\'t merge: mismatching layer sizes.');
  }

  var refNet = jsons[0];
  var mLayers = [refNet.layers[0]];
  for (var layerIdx = 1; layerIdx < refNet.layers.length; layerIdx++) {
    var mLayer = {};
    for (var neuronId in refNet.layers[layerIdx]) {
      var mNeuron = {
        bias: 0,
        weights: {}
      };

      var n = 1;
      var _iteratorNormalCompletion = true;
      var _didIteratorError = false;
      var _iteratorError = undefined;

      try {
        for (var _iterator = jsons[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
          var net = _step.value;

          var neuron = net.layers[layerIdx][neuronId];
          if (!neuron) {
            throw new Error('Can\'t merge: missing neuron [' + layerIdx + '][' + neuronId + ']');
          }

          mNeuron.bias += (neuron.bias - mNeuron.bias) / n;

          for (var weightId in neuron.weights) {
            var weight = mNeuron.weights[weightId] || 0;
            mNeuron.weights[weightId] = weight + (neuron.weights[weightId] - weight) / n;
          }

          n++;
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

      mLayer[neuronId] = mNeuron;
    }

    mLayers.push(mLayer);
  }

  refNet.layers = mLayers;
  var merged = new _neuralNetwork2.default();
  merged.fromJSON(refNet);
  return merged;
}

function equalsShallow() {
  for (var _len2 = arguments.length, arrays = Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
    arrays[_key2] = arguments[_key2];
  }

  if (!!arrays && arrays.length) {
    var len = arrays[0].length;
    for (var a = 1; a < arrays.length; a++) {
      if (arrays[a].length !== len) {
        return false;
      }
    }

    for (var i = 0; i < len; i++) {
      var item = arrays[0][i];
      for (var _a = 1; _a < arrays.length; _a++) {
        if (item !== arrays[_a][i]) {
          return false;
        }
      }
    }
  }

  return true;
}
//# sourceMappingURL=avg-nets.js.map