"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = partition;
function partition(array) {
  var partitions = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;
  var partitionSize = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 1;

  var stdSize = Math.floor(array.length / partitions);
  var step = stdSize;
  if (partitionSize <= stdSize) {
    partitionSize = stdSize;
  } else {
    step = partitionSize - stdSize;
    if (array.length / partitions > stdSize) {
      step += 1;
    }
  }

  var partitioned = [];
  var from = 0;
  for (var p = 0; p < partitions - 1; p++) {
    partitioned.push(array.slice(from, Math.min(array.length, from + partitionSize)));
    from += step;
  }
  var tailIdx = Math.min(array.length - partitionSize, from + partitionSize);
  partitioned.push(array.slice(tailIdx));

  return partitioned;
}