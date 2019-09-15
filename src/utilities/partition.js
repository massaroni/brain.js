module.exports = function partition(array, partitions = 1, partitionSize = 1) {
  if (partitionSize >= array.length) {
    const partitioned = [];
    for (let p = 0; p < partitions; p++) {
      partitioned.push(array);
    }
    return partitioned;
  }

  const stdSize = Math.floor(array.length / partitions);
  let step = stdSize;
  if (partitionSize <= stdSize) {
    partitionSize = stdSize;
  } else {
    step = (partitionSize - stdSize);
    if (array.length / partitions > stdSize) {
      step += 1;
    }
  }
  
  let partitioned = [];
  let from = 0;
  for (let p = 0; p < partitions - 1; p++) {
    partitioned.push(array.slice(from, Math.min(array.length, from + partitionSize)));
    from += step;
  }
  let tailIdx = Math.min(array.length - partitionSize, from + partitionSize);
  partitioned.push(array.slice(tailIdx));
  
  return partitioned;
}