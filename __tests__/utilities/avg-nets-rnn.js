const assert = require('assert');
const avgNetsRnnJson = require('../../src/utilities/avg-nets-rnn').avgNetsRnnJson;

describe('avg-nets-rnn', () => {

  describe('LSTM', () => {

    it('averages the weights of three nets', () => {
      const jsonA = newNet(0.5);
      const jsonB = newNet(0.1);
      const jsonC = newNet(0.6);
      const jsonActual = avgNetsRnnJson(jsonA, jsonB, jsonC);
      const jsonExpected = newNet((0.5 + 0.1 + 0.6) / 3);
      assert.deepEqual(jsonExpected, jsonActual);
    });

    it('averages the weights of many nets with high precision', () => {
      const values = [
        0.058879022977933286, 
        -0.02431313682690635, 
        -0.05476195790180191, 
        0.06875984469657391,
        0.014194271046724636,
        0.0935367434210442
      ];
      const jsons = values.map((v) => newNet(v));
      const jsonActual = avgNetsRnnJson(...jsons);
      const expectedAvg = values.reduce((v, t) => v + t, 0) / values.length;
      const jsonExpected = newNet(expectedAvg);
      assert.deepEqual(jsonExpected, jsonActual);
    });

  });

  /**
   * Generate a json blob of an RNN for testing purposes with all the same weights.
   * @param {*} weight 
   */
  function newNet(weight) {
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
