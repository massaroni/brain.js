/**
 * Utilities for generating example data for testing RNNs.
 */

const shuffle = require('./shuffle-deterministic');
const math = require('math-expression-evaluator');

module.exports = {
  makeSerieses, testTimeStep
};

function makeSeries(expression = 'sin(x)', xMin, length, xStep = 1, normalizeNumber = Math.fround) {
  const curve = [];
  for (let i = 0; i < length; i++) {
    const x = xMin + (xStep * i);
    const exp = expression.replace(/x/g, x.toString());
    const y = math.eval(exp);
    const yNormal = normalizeNumber(y);
    curve.push(yNormal);
  }

  return curve;
}

function makeSerieses(curveCount, curveLen, expression = 'sin(x)', xSpacing = 1, xStep = 1, xMin = 0, normalizeNumber = Math.fround, shuffled = true) {
  const curves = [];

  for (let i = 0; i < curveCount; i++) {
    const x = xMin + (xSpacing * i);
    curves.push(makeSeries(expression, x, curveLen, xStep));
  }

  if (shuffled) {
    shuffle(curves);
  }
  
  return curves;
}

function testTimeStep(net, serieses, verbosity = 0) {
  const suffixLen = 1;
  let errors = 0;
  let avgError = 0;

  for (let i = 0; i < serieses.length; i++) {
    const curve = serieses[i];
    const input = curve.slice(0, curve.length - suffixLen);
    const expected = curve.slice(curve.length - suffixLen);
    const predicted = net.run(input);

    if (predicted !== expected) {
      const error = Math.abs((expected - predicted) / expected);
      avgError = avgError + (error - avgError) / (errors + 1);

      if (errors < verbosity) {
        console.log('error: ' + Math.round(error * 1000) / 10 + '%');
        console.log('expected', expected);
        console.log('but was ', predicted);
        console.log('\n');
      }

      errors++;
    }
  }

  const exactMatches = serieses.length - errors;
  const avgErrorRounded = Math.round(avgError * 10000)/10000;
  const avgErrorPct = avgErrorRounded * 100;
  if (verbosity > 0) {
    console.log(`Exact Matches: ${exactMatches} out of ` + serieses.length);
    console.log(`Avg Error: ${avgErrorRounded}  (${avgErrorPct}%)`);
  }

  return {
    exactMatches,
    error: avgError
  };
}