# Emotion Classifier

Store trained model weights as `emotion_classifier.pth` in this directory. Record training details here for reproducibility.

## Suggested Metadata
- architecture: resnet50 fine-tuned on anime emotion dataset
- input resolution: 224x224
- preprocessing: face crop -> resize -> normalize
- label set: neutral, joy, anger, sadness, surprise
- training data source: describe datasets and splits
- metrics: accuracy / f1 per class
- notes: any anomalies or special handling
