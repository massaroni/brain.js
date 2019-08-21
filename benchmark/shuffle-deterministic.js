/**
 * Shuffle the array in place.
 */
module.exports = function shuffleDeterministic(array, itemToStrKey = (x) => x.toString()) {
  if (!array || !array.length) {
    return;
  }

  array.sort((lhs, rhs) => stringToIntHash(itemToStrKey(lhs)) - stringToIntHash(itemToStrKey(rhs)));
};

// thanks https://stackoverflow.com/questions/7616461/generate-a-hash-from-string-in-javascript/22429679
function stringToIntHash(s) {
  var hash = 0, i, chr;
  if (s.length === 0) return hash;
  for (i = 0; i < this.length; i++) {
    chr   = s.charCodeAt(i);
    hash  = ((hash << 5) - hash) + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
};
