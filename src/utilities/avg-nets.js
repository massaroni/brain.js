const equalsShallow = require('./eq-shallow');

/**
 * Parameter averaging, supports NeuralNetwork and NeuralNetworkGPU.
 * @param  {...any} nets 
 */
module.exports = function avgNets(...nets) {
  if (!nets || !nets.length) {
    return;
  }

  if (nets.length === 1) {
    return nets[0];
  }

  const jsons = nets.map(net => net.toJSON());

  if (!equalsShallow(jsons.map(json => json.sizes))) {
    throw new Error('Can\'t merge: mismatching layer sizes.');
  }

  let refNet = jsons[0];
  let mLayers = [refNet.layers[0]];
  for (let layerIdx = 1; layerIdx < refNet.layers.length; layerIdx++) {
    let mLayer = {};
    for (let neuronId in refNet.layers[layerIdx]) {
      let mNeuron = {
        bias: 0,
        weights: {}
      };
      
      let n = 1;
      for (let net of jsons) {
        const neuron = net.layers[layerIdx][neuronId];
        if (!neuron) {
          throw new Error('Can\'t merge: missing neuron [' + layerIdx + '][' + neuronId + ']');
        }

        mNeuron.bias += (neuron.bias - mNeuron.bias) / n;

        for (let weightId in neuron.weights) {
          const weight = mNeuron.weights[weightId] || 0;
          mNeuron.weights[weightId] = weight + (neuron.weights[weightId] - weight) / n;
        }

        n++;
      }

      mLayer[neuronId] = mNeuron;
    }

    mLayers.push(mLayer);
  }
  
  refNet.layers = mLayers;
  const netCtor = Object.getPrototypeOf(nets[0]).constructor;
  const merged = new netCtor();
  merged.fromJSON(refNet);
  return merged;
};