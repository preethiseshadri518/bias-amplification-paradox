from transformers import CLIPProcessor, CLIPModel
import torch
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
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_model = clip_model.to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
       
    
def infer_gender(image):
    text = ["A photo of a man", "A photo of a woman"]
        
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        # outputs = clip_model(**inputs)
        outputs = clip_model(**inputs.to('cuda'))
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    return probs.detach().cpu().numpy()[0,1] # returns probability female


def main():
    start_time = time.time()
    files = glob.glob(f'{image_directory}/*')

    labels, probs = [], []
    for file in files:
        img = Image.open(file)
        label = detect_faces(img, min_confidence=min_confidence, min_height=min_height, min_width=min_width)
        if label == 'filtered': # only compute gender predictions for cases with clear faces
            prob = infer_gender(img)
        else:
            prob = -1 
        labels.append(label)
        probs.append(prob) 

    df = pd.DataFrame({
        'file': files,
        'label': labels,
        'probs': probs
    })
    df.to_csv(label_filename, index=False)
    print('time:', time.time()-start_time)

    
if __name__ == '__main__':
    main() 