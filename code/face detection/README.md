# Running Face Detection and Inferring Gender
## Installing MediaPipe

To perform face detection, you need to install [mediapipe](https://developers.google.com/mediapipe/api/solutions):
```bash
pip install mediapipe
```

## Installing other Requirements

You will also need to install the following:
```bash
pip install transformers
pip install torch
pip install numpy
pip install pandas
pip install pillow
```

## Running the script
To run the face detection script, specify a directory with images (e.g. 'images/') and a CSV file to save face detection labels and gender predictions (e.g. 'labels.csv'). We only compute gender predictions for images in which we detect a clear face (i.e. labeled as 'filtered'). 
```bash
python face_detection.py image_directory label_filename
```
You may want to only consider gender predictions that have predicted probabilities within a certain threshold, so that only highly confident predictions are included (e.g. predicted probability  of female is <0.1 or >0.9) -- you can decide what this threshold should be for your specific use case.
