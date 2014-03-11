"""
Docstring spam that will be copypasted into multiple docstrings.

"""


# put some named string substitutions into a docstring
def ddec(**kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj
    return dec

