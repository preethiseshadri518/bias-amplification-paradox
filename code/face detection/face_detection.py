import mediapipe as mp
from PIL import Image
import numpy as np
import pandas as pd
import argparse
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument("image_directory", type=str, help="directory containing images to label")
parser.add_argument("label_filename", type=str, help="name of CSV file containing labels")
args = parser.parse_args()
image_directory = args.image_directory
label_filename = args.label_filename

mp_face_detection = mp.solutions.face_detection

# default values
min_confidence = 0.5
min_height = 0.1
min_width = 0.1

def detect_faces(img, min_confidence=0.5, min_height=0.1, min_width=0.1):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as face_detection:
        result = face_detection.process(np.array(img))
        if result.detections is None:
            return 'no_face'
        elif len(result.detections) > 1:
            return 'many_faces'
        else:
            height = result.detections[0].location_data.relative_bounding_box.height 
            width = result.detections[0].location_data.relative_bounding_box.width 
            if height < min_height or width < min_width:
                return 'small_face'
            else:
                return 'filtered' # images that we want to save

def main():
    start_time = time.time()
    files = glob.glob(f'{image_directory}/*')

    labels = []
    for file in files:
        img = Image.open(file)
        label = detect_faces(img, min_confidence=min_confidence, min_height=min_height, min_width=min_width)
        labels.append(label)

    df = pd.DataFrame({
        'file': files,
        'label': labels
    })
    df.to_csv(label_filename, index=False)
    print('time:', time.time()-start_time)

if __name__ == '__main__':
    main() 