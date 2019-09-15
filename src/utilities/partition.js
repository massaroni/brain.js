module.exports = function partition(array, partitions = 1, partitionSize, shuffler = shuffleCopy) {
  if (!partitionSize || (partitions * partitionSize < array.length)) {
    partitionSize = Math.ceil(array.length / partitions);
  }

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
  const gap = array.length - partitionSize;
  const spacing = Math.max(1, Math.floor(gap / (partitions - 1)));

  for (let p = 0; p < partitions - 1; p++) {
    const from = p * spacing;
    partitioned.push(array.slice(from, from + partitionSize));
  }

  const tailIdx = array.length - partitionSize;
  partitioned.push(array.slice(tailIdx));
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
