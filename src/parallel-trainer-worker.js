const brainjs = require('./index');

module.exports = function (job, callback) {
  try {
    const ctor = brainjs.get(job.netType);
    const net = new ctor();
    net.fromJSON(job.netJSON);
    const status = net.train(job.trainingData, job.trainOpts);
    const trainedNetJSON = net.toJSON();
    callback(null, {status, trainedNetJSON});
  } catch (e) {
    callback(e.toString());
  }
}