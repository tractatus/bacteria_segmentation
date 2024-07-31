import cv2
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from datetime import datetime  # Import datetime module

from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

def process_images(image_folder):
    # Load images
    X_paths = sorted(glob(os.path.join(image_folder, '*.tif')))
    X = [cv2.imread(path, -1) for path in X_paths]

    # Normalize images
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0, 1)   # normalize channels independently
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    model = StarDist2D(None, name='ecoli_dapi', basedir='models')


    for idx, img in enumerate(X):

        # Normalize the grayscale image
        img_norm = normalize(img, 1, 99.8, axis=axis_norm)
        
        # Predict instances using the StarDist model
        labels, details = model.predict_instances(img_norm)

        output_filename = f"output_{os.path.splitext(os.path.basename(X_paths[idx]))[0]}.tif"
        output_roi = f"output_{os.path.splitext(os.path.basename(X_paths[idx]))[0]}.zip"

        save_tiff_imagej_compatible(output_filename, labels, axes='YX')
        export_imagej_rois(output_roi, details['coord'])       



    # Print cell counts per image and compute average

    print(f"\nProcessing complete.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python count_cells.py <image_folder>, e.g. python prediction.py ./images/")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory.")
        sys.exit(1)

    process_images(image_folder)

if __name__ == "__main__":
    main()