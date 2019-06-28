import assert from 'assert';
import NeuralNetwork from '../../src/neural-network';
import mergeNets from '../../src/utilities/merge-nets';

describe('merge-nets', () => {

  describe('NeuralNetwork', () => {

    it('averages the weights of three nets', () => {
      const jsonA = {
        "sizes": [ 2, 2, 1 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 100,
              "weights": {
                "A": 10,
                "B": 1
              }
            },
            "1": {
              "bias": 10,
              "weights": {
                "A": 1,
                "B": 100
              }
            }
          },
          {
            "X": {
              "bias": 1,
              "weights": {
                "0": 10,
                "1": 1
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const jsonB = {
        "sizes": [ 2, 2, 1 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 200,
              "weights": {
                "A": 20,
                "B": 2
              }
            },
            "1": {
              "bias": 20,
              "weights": {
                "A": 2,
                "B": 200
              }
            }
          },
          {
            "X": {
              "bias": 2,
              "weights": {
                "0": 20,
                "1": 2
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const jsonC = {
        "sizes": [ 2, 2, 1 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 600,
              "weights": {
                "A": 60,
                "B": 6
              }
            },
            "1": {
              "bias": 60,
              "weights": {
                "A": 6,
                "B": 600
              }
            }
          },
          {
            "X": {
              "bias": 6,
              "weights": {
                "0": 60,
                "1": 6
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const jsonExpected = {
        "sizes": [ 2, 2, 1 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 300,
              "weights": {
                "A": 30,
                "B": 3
              }
            },
            "1": {
              "bias": 30,
              "weights": {
                "A": 3,
                "B": 300
              }
            }
          },
          {
            "X": {
              "bias": 3,
              "weights": {
                "0": 30,
                "1": 3
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const a = new NeuralNetwork();
      a.fromJSON(jsonA);

      const b = new NeuralNetwork();
      b.fromJSON(jsonB);

      const c = new NeuralNetwork();
      c.fromJSON(jsonC);

      const actual = mergeNets(a, b, c);
      const jsonActual = actual.toJSON();

      assert.deepEqual(jsonExpected, jsonActual);
    });

    it('averages the weights of two nets', () => {
      const jsonA = {
        "sizes": [ 2, 3, 2 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 10,
              "weights": {
                "A": 1,
                "B": 100
              }
            },
            "1": {
              "bias": 1,
              "weights": {
                "A": 100,
                "B": 10
              }
            },
            "2": {
              "bias": 100,
              "weights": {
                "A": 1,
                "B": 10
              }
            }
          },
          {
            "X": {
              "bias": 1000,
              "weights": {
                "0": 10,
                "1": 100,
                "2": 1
              }
            },
            "Y": {
              "bias": 1,
              "weights": {
                "0": 10,
                "1": 100,
                "2": 1000
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const jsonB = {
        "sizes": [ 2, 3, 2 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 70,
              "weights": {
                "A": 7,
                "B": 700
              }
            },
            "1": {
              "bias": 7,
              "weights": {
                "A": 700,
                "B": 70
              }
            },
            "2": {
              "bias": 700,
              "weights": {
                "A": 7,
                "B": 70
              }
            }
          },
          {
            "X": {
              "bias": 7000,
              "weights": {
                "0": 70,
                "1": 700,
                "2": 7
              }
            },
            "Y": {
              "bias": 7,
              "weights": {
                "0": 70,
                "1": 700,
                "2": 7000
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const jsonExpected = {
        "sizes": [ 2, 3, 2 ],
        "layers": [
          {
            "A": {},
            "B": {}
          },
          {
            "0": {
              "bias": 40,
              "weights": {
                "A": 4,
                "B": 400
              }
            },
            "1": {
              "bias": 4,
              "weights": {
                "A": 400,
                "B": 40
              }
            },
            "2": {
              "bias": 400,
              "weights": {
                "A": 4,
                "B": 40
              }
            }
          },
          {
            "X": {
              "bias": 4000,
              "weights": {
                "0": 40,
                "1": 400,
                "2": 4
              }
            },
            "Y": {
              "bias": 4,
              "weights": {
                "0": 40,
                "1": 400,
                "2": 4000
              }
            }
          }
        ],
        "outputLookup": true,
        "inputLookup": true,
        "activation": "sigmoid",
        "trainOpts": {
          "iterations": 20000,
          "errorThresh": 0.005,
          "log": false,
          "logPeriod": 10,
          "learningRate": 0.3,
          "momentum": 0.1,
          "callbackPeriod": 10,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        }
      };

      const a = new NeuralNetwork();
      a.fromJSON(jsonA);

      const b = new NeuralNetwork();
      b.fromJSON(jsonB);

      const c = mergeNets(a, b);
      const jsonC = c.toJSON();
      assert.deepEqual(jsonC, jsonExpected);
    });


    it('can have different weights for different data sets', () => {
      const trainingDataA = [
        {input: { one: 1, two: 1 }, output: { three: 1 } },
        {input: { one: 1, three: 1 }, output: { four: 1 } },
        {input: { two: 1, three: 1 }, output: { five: 1 } },
        {input: { two: 1, four: 1 }, output: { six: 1 } }
      ];
      const netA = new NeuralNetwork();
      netA.train(trainingDataA);
  
      const trainingDataB = [
        {input: { three: 1 }, output: { three: 1 } },
        {input: { one: 1 }, output: { one: 1 } },
        {input: { one: 1, four: 1 }, output: { five: 1 } },
        {input: { one: 1, two: 1 }, output: { three: 1 } },
        {input: { two: 1, three: 1 }, output: { five: 1 } },
        {input: { two: 1, four: 1 }, output: { six: 1 } },
        {input: { four: 1 }, output: { four: 1 } }
      ];
      const netB = new NeuralNetwork();
      netB.train(trainingDataB);

      assert.notDeepEqual(netA.toJSON(), netB.toJSON());
      assert.deepEqual(netA.toJSON(), netA.toJSON());
    });


  });

});
