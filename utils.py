# utils.py
import os
import glob
import pickle
import cv2

def load_images_from_folder(folder, extensions=['.jpg', '.jpeg', '.png']):
    """
    Returns a list of tuples: (path, image) for every image found recursively.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True))
    # Read images once and return tuples (path, image). If cv2.imread fails, image will be None.
    return [(path, cv2.imread(path)) for path in sorted(image_paths)]

def save_pickle(data, filename):
    """
    Save Python object to pickle file (creates dir if needed).
    """
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    """
    Load Python object from pickle file. Raises FileNotFoundError if missing.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Pickle file '{filename}' not found.")
    with open(filename, 'rb') as f:
        return pickle.load(f)