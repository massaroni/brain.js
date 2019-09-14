const equalsShallow = require('./eq-shallow');
const objectEqualsShallow = require('./obj-eq-shallow');

/**
 * Parameter averaging, supports all RNN net types.
 * @param  {...any} nets 
 */
function avgNetsRnn(...nets) {
  if (!nets || !nets.length) {
    return;
  }

  if (nets.length === 1) {
    return nets[0];
  }

  const jsons = nets.map(net => net.toJSON());
  const avgJson = avgNetsRnnJson(...jsons);
  const netCtor = Object.getPrototypeOf(nets[0]).constructor;
  const merged = new netCtor();
  merged.fromJSON(avgJson);
  return merged;
};

function avgNetsRnnJson(...jsons) {
  checkCompatibility(jsons);
  const input = avgLayer(jsons.map((j) => j.input));
  const outputConnector = avgLayer(jsons.map((j) => j.outputConnector));
  const output = avgLayer(jsons.map((j) => j.output));
  const hiddenLayers = avgHiddenLayers(jsons);
  const options = jsons[0].options; // clone this?
  const type = jsons[0].type;
  return {type, options, input, hiddenLayers, outputConnector, output};
}

function avgHiddenLayers(jsons) {
  const layerCount = (jsons[0].hiddenLayers || []).length;
  let hiddenLayersAvg = [];

  for (let i = 0; i < layerCount; i++) {
    const layer = avgHiddenLayer(jsons.map((j) => j.hiddenLayers[i]));
    hiddenLayersAvg.push(layer);
  }

  return hiddenLayersAvg;
}

function avgHiddenLayer(layers) {
  const weight = avgLayer(layers.map((l) => l.weight));
  const transition = avgLayer(layers.map((l) => l.transition));
  const bias = avgLayer(layers.map((l) => l.bias));
  return {weight, transition, bias};
}

function avgLayer(layers) {
  const rows = layers[0].rows;
  const columns = layers[0].columns;
  const weightCount = rows * columns;
  
  let weights = {};
  for (let j = 0; j < layers.length; j++) {
    for (let w = 0; w < weightCount; w++) {
      const weightAvg = weights[w] || 0;
      const weightLayer = layers[j].weights[w];
      weights[w] = weightAvg + (weightLayer - weightAvg) / (j + 1);
    }
  }

  return {rows, columns, weights};
}

function checkCompatibility(jsons) {
  const refNet = jsons[0];
  const type = refNet.type;

  for (let j = 1; j < jsons.length; j++) {
    const json = jsons[j];
    if (json.type !== type) {
      throw new Error('Mismatching nets: ' + type + ' / ' + json.type);
    }

    checkEq('inputSize', json.options, refNet.options);
    checkEq('inputRange', json.options, refNet.options);
    checkEq('outputSize', json.options, refNet.options);
  }

  const options = jsons.map((j) => j.options);
  checkArraysEq('sizes', options);
  checkArraysEq('hiddenLayers', options);
  checkArraysEq('characters', options);
  checkArraysEq('specialIndexes', options);
  checkObjectsEq('indexTable', options);
  checkObjectsEq('characterTable', options);
}

function checkArraysEq(propName, jsons) {
  checkProp(propName, equalsShallow(jsons.map(json => json[propName])));
}

function checkObjectsEq(propName, jsons) {
  checkProp(propName, objectEqualsShallow(jsons.map(json => json[propName])));
}

function checkEq(propName, lhsJson, rhsJson) {
  checkProp(propName, lhsJson[propName] === rhsJson[propName]);
}

function checkProp(propName, condition) {
  if (!condition) {
    throw new Error('Incompatible nets: mismatching ' + propName);
  }
}

module.exports = { avgNetsRnn, avgNetsRnnJson };
