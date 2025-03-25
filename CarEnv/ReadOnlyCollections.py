# Provides shallow read-only collections. These collections are not fool proof but generally help to prevent some
# common errors. You should not try to work around them.


def _readonly(self, *args, **kwargs):
    raise RuntimeError("Invalid on read-only collection")


class ReadOnlyDict(dict):
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly


class ReadOnlyList(list):
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    clear = _readonly
    extend = _readonly
    remove = _readonly
    insert = _readonly
    reverse = _readonly
