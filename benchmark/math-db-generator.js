const math = require('math-expression-evaluator');

module.exports = function (trainingPoints = 1000, expression = 'sin(x)', inputNeurons = 1, xMin, xMax, normalizeX = false) {
  if (parseInt(xMin) !== xMin || parseInt(xMax) !== xMax) {
    xMin = 0;
    xMax = trainingPoints;
  } else if (xMin >= xMax) {
    throw new Error('min/max out of range: ' + xMin + ',' + xMax);
  }

  const xSpan = Math.abs(xMax - xMin);
  const step = xSpan / trainingPoints;
  const brainTrainingSet = [];
  const buffer = [];
  for (let x = 0; brainTrainingSet.length < trainingPoints; x += step) {
    const exp = expression.replace(/x/g, x.toString());
    const y = math.eval(exp); // sin is easy
    
    if (buffer.push(y) >= inputNeurons) {
      let input = [];
      let output = [];
      for (let i = 0; i < buffer.length; i++) {
        output.push(buffer[i]);
        const xSeq = x - (buffer.length - (i + 1));
        const xNormal = normalizeX ? (xSeq - xMin) / xSpan : x;
        input.push(xNormal);
      }
      brainTrainingSet.push({input, output});
      buffer.shift();
    }
  }

  return brainTrainingSet;
};