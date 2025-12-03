import numpy as np
import os

DEFAULT_CLASSES = ['one', 'two', 'three', 'four', 'five']

def main():
    npy_path = "label_classes.npy"
    # If features_test.csv exists, try to infer classes from it
    csv_path = "features_test.csv"
    classes = None
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'label' in df.columns:
                classes = list(pd.unique(df['label']))
        except Exception:
            classes = None

    if classes is None or len(classes) == 0:
        classes = DEFAULT_CLASSES

    # Save as proper numpy .npy file
    arr = np.asarray(classes, dtype=object)
    np.save(npy_path, arr)
    print(f"Saved {npy_path} with classes: {arr}")

if __name__ == '__main__':
    main()
