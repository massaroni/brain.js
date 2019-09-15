module.exports = function partition(array, partitions = 1, partitionSize = 1, shuffler = shuffleCopy) {
  if (partitionSize >= array.length) {
    const partitioned = [];
    for (let p = 0; p < partitions; p++) {
      partitioned.push(array);
    }
    return partitioned;
  }

  const partitioned = [];

  const maxPartitions = (array.length - partitionSize) + 1;
  while (partitioned.length < partitions) {
    if (partitioned.length > 0) {
      array = shuffler(array);
    }

    const remainingPartitions = partitions - partitioned.length;
    const newPartitions = Math.min(remainingPartitions, maxPartitions);
    appendPartitions(array, newPartitions, partitionSize, partitioned);
  }

  return partitioned;
};

function appendPartitions(array, partitions, partitionSize, partitioned) {
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
  
  let from = 0;
  for (let p = 0; p < partitions - 1; p++) {
    partitioned.push(array.slice(from, Math.min(array.length, from + partitionSize)));
    from += step;
  }
  let tailIdx = Math.min(array.length - partitionSize, from + partitionSize);
  partitioned.push(array.slice(tailIdx));
  
  return partitioned;
}

function shuffleCopy(array) {
  const copy = [...array];
  for (let i = 0; i < copy.length; i++) {
    const j = Math.floor(Math.random() * copy.length);
    const swap = copy[i];
    copy[i] = copy[j];
    copy[j] = swap;
  }
  return copy;
}
