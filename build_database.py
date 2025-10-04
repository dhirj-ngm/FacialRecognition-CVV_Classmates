# build_database.py
import os
import argparse
import numpy as np
from utils import load_images_from_folder, save_pickle
from facial_recognition import get_single_face_embedding, _normalize_embedding

def build_database(dataset_dir, out_path='data/face_database.pkl', use_augment=False):
    """
    dataset_dir structure:
      dataset_dir/
         label1/
            img1.jpg
            img2.jpg
         label2/
            img1.jpg
    """
    print("Scanning dataset:", dataset_dir)
    # load all images as (path, img)
    items = load_images_from_folder(dataset_dir)
    print(f"Found {len(items)} images total (including subfolders).")
    db = {}
    # group paths by parent folder name
    for path, img in items:
        if img is None:
            print("Warning: failed to read:", path)
            continue
        label = os.path.basename(os.path.dirname(path))
        if label == '' or label is None:
            print("Warning: skipping file with no parent folder label:", path)
            continue
        emb = get_single_face_embedding(img, use_augment=use_augment, n_aug=3)
        if emb is None:
            print("Warning: no face found in:", path)
            continue
        if label not in db:
            db[label] = []
        db[label].append(emb)

    # For each label average embeddings
    db_avg = {}
    for label, emb_list in db.items():
        arr = np.vstack(emb_list)
        avg = np.mean(arr, axis=0)
        avg = _normalize_embedding(avg)
        db_avg[label] = avg
        print(f"Label {label}: images={len(emb_list)}, emb_shape={avg.shape}")

    # save
    save_pickle(db_avg, out_path)
    print("Saved database to:", out_path)
    return db_avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True, help='Path to unzipped dataset folder')
    parser.add_argument('--out', '-o', default='data/face_database.pkl', help='Output pickle path')
    parser.add_argument('--augment', action='store_true', help='Use augmentations when extracting embeddings')
    args = parser.parse_args()
    build_database(args.dataset, out_path=args.out, use_augment=args.augment)
