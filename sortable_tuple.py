class SortableTuple:
    def __init__(self, tup):
        self._tup = tuple(tup)

    def __lt__(self, other):
        if len(self._tup) != len(other._tup):
            raise ValueError("Tuples must be of same size")
        return SortableTuple._lt_tuples(self._tup, other._tup)

    def __ge__(self, other):
        return ~self.__lt__(other)

    def _lt_tuples(tup1, tup2):
        first_comp = tup1[0] < tup2[0]
        if len(tup1) > 2:  # The last element is not subject of the sort in our case
            equal_comp = tup1[0] == tup2[0]
            recursive_comp = SortableTuple._lt_tuples(tup1[1:], tup2[1:])
            # first_comp or (recursive_comp and equal_comp)
            return first_comp | (recursive_comp & equal_comp)
        else:
            return first_comp
