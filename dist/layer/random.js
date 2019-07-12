'use strict';

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Model = require('./types').Model;
var randos2D = require('../utilities/randos-2d');
var zeros2D = require('../utilities/zeros-2d');

var Random = function (_Model) {
  _inherits(Random, _Model);

  function Random(settings) {
    _classCallCheck(this, Random);

    var _this = _possibleConstructorReturn(this, (Random.__proto__ || Object.getPrototypeOf(Random)).call(this, settings));

    _this.validate();
    _this.weights = randos2D(_this.width, _this.height);
    _this.deltas = zeros2D(_this.width, _this.height);
    return _this;
  }

  _createClass(Random, [{
    key: 'predict',
    value: function predict() {
      // throw new Error(`${this.constructor.name}-predict is not yet implemented`)
    }
  }, {
    key: 'compare',
    value: function compare() {
      // throw new Error(`${this.constructor.name}-compare is not yet implemented`)
    }
  }]);

  return Random;
}(Model);

module.exports = Random;