# Bird Watch Model Setup

This guide shows you how to set up the Bird Watch TensorFlow model for enhanced bird recognition.

## Prerequisites

1. TensorFlow and Keras dependencies (run this in your virtual environment):
```bash
pip install tensorflow keras h5py
```

## Download Model Files

The Bird Watch model requires two files from the official repository:

1. Go to [Bird Watch Releases](https://github.com/Thimira/bird_watch/releases/latest)
2. Download the latest model files:
   - `final_model_*.h5` (the TensorFlow model)
   - `class_indices_*.npy` (the class mappings)

3. Rename and place them in the `models/` directory:
   - Rename `final_model_*.h5` to `final_model.h5`
   - Rename `class_indices_*.npy` to `class_indices.npy`

## Directory Structure

After setup, your directory should look like:
```
Bird Label/
├── models/
│   ├── final_model.h5
│   └── class_indices.npy
├── app.py
└── ...
```

## Verification

Once set up, the Bird Watch model will be used as the second recognition method (after iNaturalist API) in the cascade. You'll see this message in the logs:

```
🦅 Trying Bird Watch TensorFlow model...
✅ Bird Watch prediction: [Species Name] (confidence%)
```

## About the Model

The Bird Watch model:
- Based on InceptionV3 architecture
- Trained using transfer learning and fine-tuning
- Optimized specifically for bird species identification
- Created by [Thimira Amaratunga](https://github.com/Thimira)

## Alternative

If you can't download the model files, the application will automatically fall back to other recognition methods including iNaturalist API, specialized Hugging Face models, YOLOv5, and local fallbacks.
