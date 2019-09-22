const assert = require('assert');
const avgNetsRnnJson = require('../../src/utilities/avg-nets-rnn').avgNetsRnnJson;
const brain = require('../../src');
const LSTMTimeStep = brain.recurrent.LSTMTimeStep;

describe('avg-nets-rnn', () => {

  describe('LSTMTimeStep', () => {

    it('averages two nets, smoke test', () => {
      const trainingData = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
      ];

      const net = new LSTMTimeStep();
      net.train(trainingData, {iterations: 1, log: false});
      const net2 = new LSTMTimeStep();
      net2.train(trainingData.slice(1), {iterations: 1, log: false});

      const avg = net.avg(net2);
      const prediction = avg.run([1, 2]);
      assert.equal(true, prediction < 1 && prediction > -1);

      const avgJson = avg.toJSON();
      const netJson = net.toJSON();
      const net2Json = net2.toJSON();

      const matrix = netJson.hiddenLayers[0].weight ? 'weight' : 'forgetMatrix';
      assert.equal(
        Math.fround((netJson.hiddenLayers[0][matrix].weights[0] + net2Json.hiddenLayers[0][matrix].weights[0]) / 2),
        avgJson.hiddenLayers[0][matrix].weights[0]
      );
    });

  });

  // describe('LSTM', () => {

  //   it('averages the weights of three nets', () => {
  //     const jsonA = newLstmJson(0.5);
  //     const jsonB = newLstmJson(0.1);
  //     const jsonC = newLstmJson(0.6);
  //     const jsonActual = avgNetsRnnJson(jsonA, jsonB, jsonC);
  //     const jsonExpected = newLstmJson((0.5 + 0.1 + 0.6) / 3);
  //     assert.deepEqual(jsonExpected, jsonActual);
  //   });

  //   it('averages the weights of many nets with high precision', () => {
  //     const values = [
  //       0.058879022977933286, 
  //       -0.02431313682690635, 
  //       -0.05476195790180191, 
  //       0.06875984469657391,
  //       0.014194271046724636,
  //       0.0935367434210442
  //     ];
  //     const jsons = values.map((v) => newLstmJson(v));
  //     const jsonActual = avgNetsRnnJson(...jsons);
  //     const expectedAvg = values.reduce((v, t) => v + t, 0) / values.length;
  //     const jsonExpected = newLstmJson(expectedAvg);
  //     assert.deepEqual(jsonExpected, jsonActual);
  //   });

  // });

  /**
   * Generate a json blob of an RNN for testing purposes with all the same weights.
   * @param {*} weight 
   */
  function newLstmJson(weight) {
    return {
      "type": "LSTM",
      "options": {
        "inputSize": 4,
        "inputRange": 4,
        "hiddenLayers": [
          2,
          2
        ],
        "outputSize": 4,
        "decayRate": 0.999,
        "smoothEps": 1e-8,
        "regc": 0.000001,
        "clipval": 5,
        "maxPredictionLength": 100,
        "dataFormatter": {
          "values": [
            "0+0=0",
            "0+1=1",
            "1+0=1"
          ],
          "indexTable": {
            "0": 0,
            "1": 3,
            "+": 1,
            "=": 2
          },
          "characterTable": {
            "0": "0",
            "1": "+",
            "2": "=",
            "3": "1"
          },
          "characters": [
            "0",
            "+",
            "=",
            "1"
          ],
          "specialIndexes": []
        }
      },
      "input": {
        "rows": 5,
        "columns": 4,
        "weights": {
          "0": weight,
          "1": weight,
          "2": weight,
          "3": weight,
          "4": weight,
          "5": weight,
          "6": weight,
          "7": weight,
          "8": weight,
          "9": weight,
          "10": weight,
          "11": weight,
          "12": weight,
          "13": weight,
          "14": weight,
          "15": weight,
          "16": weight,
          "17": weight,
          "18": weight,
          "19": weight
        }
      },
      "hiddenLayers": [
        {
          "weight": {
            "rows": 2,
            "columns": 4,
            "weights": {
              "0": weight,
              "1": weight,
              "2": weight,
              "3": weight,
              "4": weight,
              "5": weight,
              "6": weight,
              "7": weight
            }
          },
          "transition": {
            "rows": 2,
            "columns": 2,
            "weights": {
              "0": weight,
              "1": weight,
              "2": weight,
              "3": weight
            }
          },
          "bias": {
            "rows": 2,
            "columns": 1,
            "weights": {
              "0": weight,
              "1": weight
            }
          }
        },
        {
          "weight": {
            "rows": 2,
            "columns": 2,
            "weights": {
              "0": weight,
              "1": weight,
              "2": weight,
              "3": weight
            }
          },
          "transition": {
            "rows": 2,
            "columns": 2,
            "weights": {
              "0": weight,
              "1": weight,
              "2": weight,
              "3": weight
            }
          },
          "bias": {
            "rows": 2,
            "columns": 1,
            "weights": {
              "0": weight,
              "1": weight
            }
          }
        }
      ],
      "outputConnector": {
        "rows": 5,
        "columns": 2,
        "weights": {
          "0": weight,
          "1": weight,
          "2": weight,
          "3": weight,
          "4": weight,
          "5": weight,
          "6": weight,
          "7": weight,
          "8": weight,
          "9": weight
        }
      },
      "output": {
        "rows": 5,
        "columns": 1,
        "weights": {
          "0": weight,
          "1": weight,
          "2": weight,
          "3": weight,
          "4": weight
        }
      }
    };
  }

});
