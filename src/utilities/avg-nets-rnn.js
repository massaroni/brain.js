const { aggNetsRnn, aggNetsRnnJson } = require('./aggregate-rnn');

/**
 * Parameter averaging, supports all RNN net types.
 * @param  {...any} nets 
 */
function avgNetsRnn(...nets) {
  return aggNetsRnn(avgFactory, ...nets);
};

function avgNetsRnnJson(...jsons) {
  return aggNetsRnnJson(avgFactory, ...jsons);
}

class Averager {
  constructor() {
    this.avg = 0;
    this.count = 0;
  }

  agg(n) {
    this.avg = this.avg + (n - this.avg) / (this.count++ + 1);
  }

  build() {
    return this.avg;
  }
}

function avgFactory() {
  return new Averager();
}

module.exports = { avgNetsRnn, avgNetsRnnJson };
