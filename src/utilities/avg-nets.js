import NeuralNetwork from '../neural-network';

/**
 * Parameter averaging, supports NeuralNetwork and NeuralNetworkGPU.
 * @param  {...any} nets 
 */
export default function avgNets(...nets) {
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
  const merged = new NeuralNetwork();
  merged.fromJSON(refNet);
  return merged;
}

function equalsShallow(...arrays) {
  if (!!arrays && arrays.length) {
    const len = arrays[0].length;
    for (let a = 1; a < arrays.length; a++) {
      if (arrays[a].length !== len) {
        return false;
      }
    }

    for (let i = 0; i < len; i++) {
      const item = arrays[0][i];
      for (let a = 1; a < arrays.length; a++) {
        if (item !== arrays[a][i]) {
          return false;
        }
      }
    }
  }

  return true;
}