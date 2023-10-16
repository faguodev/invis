'''
Created on Jan 6, 2014

'''

class comparable(object):
    
    
    def _compare(self, other, method):
        try:
            return method(self._cmp_key(), other._cmp_key())
        except (AttributeError, TypeError):
            return NotImplemented


    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)


    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)


    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)


    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)


    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)


    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
