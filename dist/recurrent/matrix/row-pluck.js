"use strict";

/**
 * @param {Matrix} product
 * @param {Matrix} left
 * @param {Number} rowPluckIndex
 */
module.exports = function rowPluck(product, left, rowPluckIndex) {
  var columns = left.columns;

  var rowBase = columns * rowPluckIndex;
  for (var column = 0; column < columns; column++) {
    product.weights[column] = left.weights[rowBase + column];
    product.deltas[column] = 0;
  }
};