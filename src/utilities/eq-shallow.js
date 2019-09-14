module.exports = function equalsShallow(...arrays) {
  if (!!arrays && arrays.length) {
    const len = arrays[0].length;
    for (let a = 1; a < arrays.length; a++) {
      if (arrays[a].length !== len) {
        return false;
      }
    }

    for (let i = 0; i < len; i++) {
      const item = arrays[0][i];
      for (let a = 1; a < arrays.length; a++) {
        if (item !== arrays[a][i]) {
          return false;
        }
      }
    }
  }

  return true;
};