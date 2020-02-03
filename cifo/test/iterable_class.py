class my:
    def __init__(self, list=[1, 3, 5, 7, 9]):
        self._list = list
        # iterable index
        self._index = 0

    # iterable methods __iter__
    def __iter__(self):
        return self

    # iterable methods : __next__
    def __next__(self):
        """'Returns the next value from team object's lists """
        if self._index < len(self._list):
            result = self._list[self._index]
            self._index += 1
            return result
        # End of Iteration

        else:
            self._index = 0
        raise StopIteration


my1 = my()

i = 0
for m in my1:
    i += 1
    print(m)
    if i == 2:
        break

print("--------------------")
for m in my1:
    print(m)
