'use strict';

var brainjs = require('./index').default;

module.exports = function (job, callback) {
  try {
    var net = new brainjs[job.netType]();
    net.fromJSON(job.netJSON);
    var status = net.train(job.trainingData, job.trainOpts);
    var trainedNetJSON = net.toJSON();
    callback(null, { status: status, trainedNetJSON: trainedNetJSON });
  } catch (e) {
    callback(e.toString());
  }
};
//# sourceMappingURL=parallel-trainer-worker.js.map