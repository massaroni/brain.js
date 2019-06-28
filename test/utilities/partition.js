import assert from 'assert';
import partition from '../../src/utilities/partition';

describe('partition', () => {

  it('can partition an even-len array without overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3, 2);
    assert.deepStrictEqual(actual, [[1, 2], [3, 4], [5, 6]]);
  });

  it('can partition an even-len array with overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6], 3, 3);
    assert.deepStrictEqual(actual, [[1, 2, 3], [3, 4, 5], [5, 6]]);
  });

  it('can partition an odd-len array with overlap and the remainder at the end', () => {
    const actual = partition([1, 2, 3, 4, 5], 3, 2);
    assert.deepStrictEqual(actual, [[1, 2], [2, 3], [3, 4, 5]]);
  });

  it('can partition an odd-len array with congruent overlap', () => {
    const actual = partition([1, 2, 3, 4, 5, 6, 7], 3, 3);
    assert.deepStrictEqual(actual, [[1, 2, 3], [3, 4, 5], [5, 6, 7]]);
  });

});
