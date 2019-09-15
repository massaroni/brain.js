const { aggNetsRnn, aggNetsRnnJson } = require('./aggregate-rnn');

function multNetRnn(multiplier, ...nets) {
  checkInputs(nets);
  return aggNetsRnn(() => new ScalarMultiplier(multiplier), ...nets);
}

function multNetRnnJson(multiplier, ...jsons) {
  checkInputs(jsons);
  return aggNetsRnnJson(() => new ScalarMultiplier(multiplier), ...jsons);
}

function checkInputs(inputs) {
  if (inputs.length !== 1) {
    throw new Error('Scalar multiplication requires exactly one input RNN.');
  }
}

class ScalarMultiplier {
  constructor(multiplier) {
    this.multiplier = multiplier;
  }

  agg(n) {
    this.value = n * this.multiplier;
  }

  build() {
    return this.value;
  }
}

module.exports = { multNetRnn, multNetRnnJson };
