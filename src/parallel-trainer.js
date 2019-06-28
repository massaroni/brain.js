import mergeNets from './utilities/merge-nets';
import partition from './utilities/partition';

/**
 * Ensemble averaging, via simple parameter averaging.
 */
export default class ParallelTrainer {
  constructor(trainOptions = {}) {
    this.trainOptions = trainOptions;
  }

  async train(data, net, trainOptionsOverrides = {}) {
    const startMs = Date.now();
    const trainOptions = Object.assign({}, this.trainOptions, trainOptionsOverrides);
    const pOptions = trainOptions.parallel || {};
    const threadCount = pOptions.threads || 1;
    const minPartitionSize = Math.ceil(data.length / threadCount);
    const partitionSize = Math.max(Math.ceil(pOptions.partitionSize || 1), minPartitionSize);
    const partitions = partition(data, threadCount, partitionSize);
    const NetCtor = Object.getPrototypeOf(net).constructor;
    const maxEpochs = pOptions.maxEpochs || 1000;
    const errorThresh = trainOptions.errorThresh || NetCtor.trainDefaults.errorThresh;
    
    let peerTrainOptions = Object.assign({}, trainOptions);
    delete peerTrainOptions.parallel;
    delete peerTrainOptions.callback;
    delete peerTrainOptions.log;
    
    net.train([data[0]], {errorThresh: 0.9, iterations: 1}); // initialize weights
    let globalWeights = net.toJSON();
  
    let error = 1;
    let epochs = 0;
    let iterations = 0;

    while (epochs < maxEpochs && error >= errorThresh) {
      let promises = [];
      let peers = [];

      for (let t = 0; t < threadCount; t++) {
        let peer = new NetCtor();
        peer.fromJSON(globalWeights);
        peers.push(peer);

        if (pOptions.syncMode === true) {
          promises.push(Promise.resolve(peer.train(partitions[t], peerTrainOptions)));
        } else {
          promises.push(peer.trainAsync(partitions[t], peerTrainOptions));
        }
      }

      const statuses = await Promise.all(promises);
      let worstError = 0;
      for (let t = threadCount - 1; t >= 0; t--) {
        const peer = peers[t];
        const partitionIdx = (t === 0 ? threadCount : t) - 1;
        const result = peer.test(partitions[partitionIdx][0]);
        worstError = Math.max(result.error, worstError);
        iterations += statuses[t].iterations;
      }
      error = worstError;
      epochs++;

      globalWeights = mergeNets(...peers).toJSON();
    }
  
    net.fromJSON(globalWeights);
    const endMs = Date.now();
    const elapsedMs = endMs - startMs;
    return {error, iterations, epochs, elapsedMs};
  }

}
