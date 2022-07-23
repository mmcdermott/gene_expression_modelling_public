import pickle, os

def depickle(path, base=os.path.join(os.path.dirname(__file__), 'data')):
    with open(os.path.join(base, path), mode='rb') as f: return pickle.load(f)
