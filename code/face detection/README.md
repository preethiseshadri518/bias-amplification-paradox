# Running Face Detection
## Installing MediaPipe

To perform face detection, you need to install [mediapipe](https://developers.google.com/mediapipe/api/solutions):
```bash
pip install mediapipe
```

## Installing other Requirements

You will also need to install the following:
```bash
pip install numpy
pip install pandas
pip install pillow
```

## Running the script
To run the face detection script, specify a directory with images (e.g. 'images/') and a CSV file to save face detection labels (e.g. 'labels.csv')
```bash
python face_detection.py image_directory label_filename
```
