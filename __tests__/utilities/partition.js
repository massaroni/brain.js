const assert = require('assert');
const partition = require('../../src/utilities/partition');

describe('partition', () => {

  it('would return the full set for full partition sizes', () => {
    const fullSet = [1, 2, 3, 4, 5, 6];
    const actual = partition(fullSet, 4, 6);
    assert.deepStrictEqual(actual, [fullSet, fullSet, fullSet, fullSet]);
  });

  it('would return the full set for oversized partition sizes', () => {
    const fullSet = [1, 2, 3, 4, 5, 6];
    const actual = partition(fullSet, 4, 60);
    assert.deepStrictEqual(actual, [fullSet, fullSet, fullSet, fullSet]);
  });

  it('can partition an even-len array without overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3, 2);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [5, 6]]);
  });

  it('can partition an even-len array with overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3, 3);
    assert.deepStrictEqual(actual, [[1, 2, 3], [2, 3, 4], [4, 5, 6]]);
  });

  it('can partition an odd-length array with some necessary overlap', () => {
    const actual = partition([1, 2, 3, 4, 5], 3, 2);
    assert.deepStrictEqual(actual, [[1, 2], [2, 3], [4, 5]]);
  });

  it('can partition an odd-len array with congruent overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6, 7], 3, 3);
    assert.deepStrictEqual(actual, [[1, 2, 3], [3, 4, 5], [5, 6, 7]]);
    const actual2 = partition([1, 2, 3, 4, 5, 6, 7, 8, 9], 4, 3);
    assert.deepStrictEqual(actual2, [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]]);
    const actual3 = partition([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3);
    assert.deepStrictEqual(actual3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  });

  it('can fall back to the minimum partition size if the requested partition size is too small to cover the whole array', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3, 1);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [5, 6]]);
  });

  it('can fall back to the minimum partition size by default', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [5, 6]]);
  });

  it('can partition overlapping segments', () => {
    const actual = partition(['a','b','c','d'], 2, 3);
    assert.deepStrictEqual(actual, [['a','b','c'],['b','c','d']]);
  });

  it('can partition duplicate overlapping segments', () => {
    const actual = partition(['a','b','c','d'], 4, 3, (a) => a);
    assert.deepStrictEqual(actual, [
      ['a','b','c'],
      ['b','c','d'],
      ['a','b','c'],
      ['b','c','d'],
    ]);
  });

  it('can partition duplicate overlapping segments with shuffling', () => {
    const actual = partition(['a','b','c','d'], 4, 3);
    assert.deepStrictEqual(actual.slice(0, 2), [
      ['a','b','c'],
      ['b','c','d']
    ]);
    assert.equal(actual.length, 4);
    assert.equal(actual[2].length, 3);
    assert.equal(actual[3].length, 3);
    assert.equal(new Set(actual[2]).size, 3);
    assert.equal(new Set(actual[3]).size, 3);
    assert.equal(new Set(actual[2].concat(actual[3])).size, 4);
  });

  it('can partition larger overlapping segments', () => {
    const fullSet = arr(0, 100);
    const actual = partition(fullSet, 6, 40, (a) => a);
    assert.deepStrictEqual(actual, [
      arr(0, 40),
      arr(12, 40),
      arr(24, 40),
      arr(36, 40),
      arr(48, 40),
      arr(60, 40),
    ]);
  });

  function arr(n, length) {
    let numbers = [];
    for (let i = 0; i < length; i++) {
      numbers.push(n + i);
    }
    return numbers;
  }
});
