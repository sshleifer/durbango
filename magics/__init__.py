from .exceptor import Exceptor

def load_ipython_extension(ipython):
    ipython.register_magics(Exceptor)
