'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Internal = require('./types').Internal;
var zeros2D = require('../utilities/zeros-2d');

var RecurrentConnection = function (_Internal) {
  _inherits(RecurrentConnection, _Internal);

  function RecurrentConnection() {
    _classCallCheck(this, RecurrentConnection);

    return _possibleConstructorReturn(this, (RecurrentConnection.__proto__ || Object.getPrototypeOf(RecurrentConnection)).apply(this, arguments));
  }

  _createClass(RecurrentConnection, [{
    key: 'setLayer',
    value: function setLayer(layer) {
      this.layer = layer;
    }
  }, {
    key: 'predict',
    value: function predict() {
      // throw new Error(`${this.constructor.name}-predict is not yet implemented`)
    }
  }, {
    key: 'compare',
    value: function compare() {
      // throw new Error(`${this.constructor.name}-compare is not yet implemented`)
    }
  }, {
    key: 'learn',
    value: function learn() {
      this.layer.deltas = zeros2D(this.width, this.height);
    }
  }, {
    key: 'setupKernels',
    value: function setupKernels() {
      // throw new Error(
      //   `${this.constructor.name}-setupKernels is not yet implemented`
      // )
    }
  }, {
    key: 'reuseKernels',
    value: function reuseKernels() {
      // throw new Error(
      //   `${this.constructor.name}-reuseKernels is not yet implemented`
      // )
    }
  }, {
    key: 'width',
    get: function get() {
      return this.layer.width;
    },
    set: function set(value) {
      throw new Error(this.constructor.name + '-width is not yet implemented');
    }
  }, {
    key: 'height',
    get: function get() {
      return this.layer.height;
    },
    set: function set(value) {
      throw new Error(this.constructor.name + '-height is not yet implemented');
    }
  }, {
    key: 'deltas',
    get: function get() {
      return this.layer.deltas;
    },
    set: function set(deltas) {
      this.layer.deltas = deltas;
    }
  }, {
    key: 'weights',
    get: function get() {
      return this.layer.weights;
    },
    set: function set(weights) {
      this.layer.weights = weights;
    }
  }]);

  return RecurrentConnection;
}(Internal);

module.exports = RecurrentConnection;