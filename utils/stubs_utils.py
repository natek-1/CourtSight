import os

import pickle


def save_stub(stub_path, object):
    
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))
    
    with open(stub_path, "wb") as file:
        pickle.dump(object, file)


def read_stub(stub_path, read_from_stub):
    
    if read_from_stub and stub_path is not None and os.path.exists(os.path.dirname(stub_path)):
        with open(stub_path, "rb") as file:
            return pickle.load(file)
    return None