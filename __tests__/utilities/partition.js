import assert from 'assert';
import partition from '../../src/utilities/partition';

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

  it('can partition an odd-len array with overlap and the remainder at the end', () => {
    const actual = partition([1, 2, 3, 4, 5], 3, 2);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [4, 5]]);
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

  it('can fall back to the minimum partition size if the requested partition size by default', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [5, 6]]);
  });

  it('can partition overlapping segments', () => {
    const actual = partition(['a','b','c','d'], 2, 3);
    assert.deepStrictEqual(actual, [['a','b','c'],['b','c','d']]);
  });

});
