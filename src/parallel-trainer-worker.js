const brainjs = require('./index').default;

module.exports = function (job, callback) {
  try {
    const net = new brainjs[job.netType]();
    net.fromJSON(job.netJSON);
    const status = net.train(job.trainingData, job.trainOpts);
    const trainedNetJSON = net.toJSON();
    callback(null, {status, trainedNetJSON});
  } catch (e) {
    callback(e.toString());
  }
}