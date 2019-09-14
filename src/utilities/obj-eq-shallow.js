module.exports = function objectEqualsShallow(...objects) {
  if (!!objects && objects.length > 1) {
    const allPropNames = objects.map((o) => Object.getOwnPropertyNames(o));
    const lhsPropNames = allPropNames[0];

    for (let i = 1; i < allPropNames.length; i++) {
      const rhsPropNames = allPropNames[i];
      if (rhsPropNames.length !== lhsPropNames.length) {
        return false;
      }

      for (var p = 0; p < lhsPropNames.length; p++) {
        const name = lhsPropNames[i];

        if (lhsPropNames[name] !== rhsPropNames[name]) {
          return false;
        }
      }
    }
  }

  return true;
};