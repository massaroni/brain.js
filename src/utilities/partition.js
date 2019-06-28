export default function partition(array, partitions = 1, partitionSize = 1) {
  const stdSize = Math.floor(array.length / partitions);
  if (partitionSize < stdSize) {
    partitionSize = stdSize;
  }
  const step = partitionSize - (partitionSize - stdSize);

  let partitioned = [];
  let from = 0;
  for (let p = 0; p < partitions - 1; p++) {
    partitioned.push(array.slice(from, Math.min(array.length, from + partitionSize)));
    from += step;
  }
  partitioned.push(array.slice(from, array.length));
  
  return partitioned;
}