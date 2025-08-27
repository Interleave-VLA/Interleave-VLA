import numpy as np

def decode_language_table(inst):
    """Utility to decode encoded language instruction"""
    inst = inst.numpy()
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

def decode(inst):
    return inst.numpy().decode('utf-8')