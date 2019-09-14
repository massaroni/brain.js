const { aggNetsRnn, aggNetsRnnJson } = require('./aggregate-rnn');

function subtractNetsRnn(...nets) {
  return aggNetsRnn(subtractFactory, ...nets);
};

function subtractNetsRnnJson(...jsons) {
  return aggNetsRnnJson(subtractFactory, ...jsons);
}

class Subtractor {
  constructor() {
    this.value = 0;
    this.count = 0;
  }

  agg(n) {
    if (this.count++ < 1) {
      this.value = n;
    } else {
      this.value -= n;
    }
  }

  build() {
    return this.value;
  }
}

function subtractFactory() {
  return new Subtractor();
}

module.exports = { subtractNetsRnn, subtractNetsRnnJson };
