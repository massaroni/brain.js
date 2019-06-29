import NeuralNetwork from './neural-network';

module.exports = function (job, callback) {
  if (job.netType !== 'NeuralNetwork') {
    return callback('Unsupported net type: ' + job.netType);
  }

  try {
    const net = new NeuralNetwork();
    net.fromJSON(job.netJSON);
    const status = net.train(job.trainingData, job.trainOpts);
    const trainedNetJSON = net.toJSON();
    callback(null, {status, trainedNetJSON});
  } catch (e) {
    callback(e.toString());
  }
}