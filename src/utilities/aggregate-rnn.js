const equalsShallow = require('./eq-shallow');
const objectEqualsShallow = require('./obj-eq-shallow');

/**
 * Aggregate nodes among a set of RNNs.
 * This could be used for averaging, multiplying, subtracting nets, etc.
 * This aggregates corresponding weights across all given RNNs.
 * RNNs must have matching architecture, or else this throws an error.
 * 
 * @param {*} aggregatorFactory callback function that returns a new aggregator
 * @param  {...any} nets RNN instances
 */
function aggNetsRnn(aggregatorFactory, ...nets) {
  if (!nets || !nets.length) {
    return;
  }

  if (nets.length === 1) {
    return nets[0];
  }

  const jsons = nets.map(net => net.toJSON());
  const avgJson = aggNetsRnnJson(aggregatorFactory, ...jsons);
  const netCtor = Object.getPrototypeOf(nets[0]).constructor;
  const aggregated = new netCtor();
  aggregated.fromJSON(avgJson);
  return aggregated;
};

function aggNetsRnnJson(aggregatorFactory, ...jsons) {
  checkCompatibility(jsons);
  const input = aggLayer(aggregatorFactory, jsons.map((j) => j.input));
  const outputConnector = aggLayer(aggregatorFactory, jsons.map((j) => j.outputConnector));
  const output = aggLayer(aggregatorFactory, jsons.map((j) => j.output));
  const hiddenLayers = aggHiddenLayers(aggregatorFactory, jsons);
  const options = jsons[0].options; // clone this?
  const type = jsons[0].type;
  return {type, options, input, hiddenLayers, outputConnector, output};
}

function aggHiddenLayers(aggregatorFactory, jsons) {
  const layerCount = (jsons[0].hiddenLayers || []).length;
  let hiddenLayersAvg = [];

  for (let i = 0; i < layerCount; i++) {
    const layer = aggHiddenLayer(aggregatorFactory, jsons.map((j) => j.hiddenLayers[i]));
    hiddenLayersAvg.push(layer);
  }

  return hiddenLayersAvg;
}

function aggHiddenLayer(aggregatorFactory, layers) {
  const weight = aggLayer(aggregatorFactory, layers.map((l) => l.weight));
  const transition = aggLayer(aggregatorFactory, layers.map((l) => l.transition));
  const bias = aggLayer(aggregatorFactory, layers.map((l) => l.bias));
  return {weight, transition, bias};
}

function aggLayer(aggregatorFactory, layers) {
  const rows = layers[0].rows;
  const columns = layers[0].columns;
  const weightCount = rows * columns;
  
  let weights = {};
  for (let w = 0; w < weightCount; w++) {
    const aggregator = aggregatorFactory();
    for (let j = 0; j < layers.length; j++) {
      aggregator.agg(layers[j].weights[w]);
    }
    weights[w] = aggregator.build();
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

module.exports = { aggNetsRnn, aggNetsRnnJson };
