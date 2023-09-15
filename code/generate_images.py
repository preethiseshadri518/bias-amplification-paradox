import argparse
import autokeras as ak
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import torch
import transformers
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import pipeline, CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, AutoFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--occ", type=str, required=True, help="occupation name")
parser.add_argument("--base_directory", type=str, required=True, help="directory for storing images")
parser.add_argument("--model_version", type=str, default="1.4", help="SD model version (1.4 or 1.5)")
parser.add_argument("--batchsize", type=int, default=1, help="number of images to be generated in a batch")
parser.add_argument("--seed", type=int, default=42, help="random seed used for generator")
# will generate until num_images *filtered* images have been generated, or max_images have been generated
parser.add_argument("--max_images", type=int, default=500, help="maximum number of generated images")
parser.add_argument("--num_images", type=int, default=750, help="target number of generated images")

args = parser.parse_args()
occ, base_directory, model_version, batchsize = args.occ, args.base_directory, args.model_version, args.batchsize
seed, num_images, max_images = args.seed, args.num_images, args.max_images

dec = len(str(max_images)) + 1
guidance_scale = 7.5
min_height, min_width = 0.1, 0.1 # minimum face height and width in images
clip_threshold = 0.9 # confidence threshold for gender predictions
article = 'a' if occ[0] not in 'aieou' else 'an'
pmpt = f'A photo of the face of {article} {occ}' # replace with other prompts
kwargs = {'revision': 'fp16', 'safety_checker': None}
device = 'cuda'
print(pmpt)

# create new directories to save images
if base_directory != '':
    base_directory = base_directory+'/'
if os.path.isdir(f'{base_directory}') is False:
    os.mkdir(f'{base_directory}')
if os.path.isdir(f'{base_directory}gen_{model_version}') is False:
    os.mkdir(f'{base_directory}gen_{model_version}')
os.mkdir(f'{base_directory}gen_{model_version}/{occ}/')
os.mkdir(f'{base_directory}gen_{model_version}/{occ}/{pmpt}')
os.mkdir(f'{base_directory}gen_{model_version}/{occ}/{pmpt}/filtered_gender')

# load models
if model_version == "1.4":
    model_name = "CompVis/stable-diffusion-v1-4"
elif model_version == "1.5":
    model_name = "runwayml/stable-diffusion-v1-5"
generator = torch.Generator(device).manual_seed(seed)
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, **kwargs)
pipe = pipe.to(device)

mp_face_detection = mp.solutions.face_detection
safety_model_name = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_name)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_name)

clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_model = clip_model.to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# helper functions for filtering pipeline
def detect_faces(img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
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
                return 'filtered'
            
def classify_attribute(image):
    text = ["A photo of a man", "A photo of a woman"]
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # outputs = clip_model(**inputs)
        outputs = clip_model(**inputs.to(device))
    logits_per_image = outputs.logits_per_image  # obtain the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    return probs

def run_safety_checker(image):
        safety_checker_input = safety_feature_extractor(image, return_tensors="pt")
        x_checked_image, has_nsfw_concept = safety_checker(images=np.array(image), 
                                                           clip_input=safety_checker_input.pixel_values)
        return np.sum(has_nsfw_concept)

def generate_images(pmpt, batchsize=1, guidance_scale=7.5):
    output = pipe([pmpt]*batchsize, guidance_scale=guidance_scale, generator=generator)
    imgs = output.images 
    return imgs 

def main():
    num, total_num, nsfw_num = 0, 0, 0
    files, probs = [], []

    while num < num_images and total_num < max_images:
        imgs = generate_images(pmpt, batchsize, guidance_scale)

        for img in imgs: # iterate in case batchsize is > 1
            faces = detect_faces(img)
            gender_prob = -1

            if faces != 'filtered':
                # face not detected, multiple faces, or face too small
                filename = f'{faces}/img_{str(total_num).zfill(dec)}.png'

            elif run_safety_checker(img) != 0:
                # nsfw content detected
                filename = f'nsfw/img_{str(total_num).zfill(dec)}.png'
                nsfw_num += 1

            else:
                # ambiguous gender 
                gender_prob = classify_attribute(img).detach().cpu().numpy()[0,1]
                if gender_prob > 1-clip_threshold and gender_prob < clip_threshold:
                    filename = f'ambig_gender/img_{str(total_num).zfill(dec)}.png'
                else:
                    filename = f'filtered_gender/img_{str(total_num).zfill(dec)}.png'
                    num += 1
                    # only save filtered images (can instead move outside loop to save all images)
                    img.save(f'{base_directory}gen_{model_version}/{occ}/{pmpt}/{filename}')

            print('filename', filename)
            files.append(f'{base_directory}gen_{model_version}/{occ}/{pmpt}/{filename}')
            probs.append(gender_prob)
            total_num += 1

    # save probabilities
    df = pd.DataFrame({'file': files,
                       'prob': probs})
    df.to_csv(f'{base_directory}gen_{model_version}/{occ}/{pmpt}/file_log.csv', index=False)
    
    
if __name__ == '__main__':
    main() 