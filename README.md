
# Blood Cell Detection Using YOLOv8

This project is the final project for the Digital Image Processing (DIP) course at the Libyan Academy of Postgraduate Studies. The aim is to detect and classify blood cells into white blood cells (WBCs) and red blood cells (RBCs) using the YOLOv8 object detection model.

## Overview

The project involves:
- Using a dataset of 100 blood smear images labeled with bounding boxes for RBCs and WBCs.
- Splitting the dataset randomly into training and evaluation sets.
- Converting the annotations into YOLOv8 format.
- Training a YOLOv8 pre-trained model on the dataset.
- Achieving an overall accuracy of 97% in classifying blood cells.

## Dataset

The dataset used in this project is sourced from Dr. Abdüssamet Aslan's repository. Special thanks to Dr. Aslan for providing the dataset. You can find the original dataset [here](https://github.com/draaslan/blood-cell-detection-dataset).

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- [Ultralytics YOLO](https://github.com/ultralytics/yolov8)

You can install the required packages using:

```bash
pip install ultralytics
```

### Project Structure

```
.
├── blood-dataset
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── val
│   │   ├── images
│   │   └── labels
│   └── test_images
├── runs
│   └── detect
│       └── train14
│           └── weights
│               └── best.pt
├── README.md
└── detect.py
```

### Training the Model

1. Prepare the dataset in YOLOv8 format with annotations.
2. Train the YOLOv8 model using the prepared dataset.

```python
from ultralytics import YOLO

# Load the YOLO model and prepare for training
model = YOLO('yolov8s.pt')

# Train the model
model.train(data='path/to/data.yaml', epochs=100, imgsz=640)
```

### Using the Trained Model

To use the trained model for making predictions on new images, follow these steps:

1. Ensure your model weights are saved at `./runs/detect/train14/weights/best.pt`.
2. Use the following code to perform object detection on a test image:

```python
from ultralytics import YOLO

# Instantiate YOLO model and load weights
model = YOLO('./runs/detect/train14/weights/best.pt')

# Specify the test image path
image_path = './blood-dataset/test_images/pool3.jpg'

try:
    # Perform object detection on the test image
    results = model(image_path, conf=0.2469)
    
    print(f"Results type: {type(results)}")  # Print the type of results returned
    print(f"Number of results: {len(results)}")  # Print the number of items in results list
    
    # Iterate through each Results object
    for idx, result in enumerate(results):
        print(f"Processing Result {idx + 1}:")
        
        # Visualize the detection results
        print("Visualizing detection results...")
        result.show()  # Show annotated results to screen
        print("Detection results visualized.")
        
        # Access the detection metrics (optional)
        print("Accessing detection metrics...")
        if result.boxes is not None and result.probs is not None:
            for bbox, conf in zip(result.boxes, result.probs):
                print(f"Bounding Box: {bbox}, Confidence: {conf}")
            print("Detection metrics accessed successfully.")
        else:
            print("No bounding boxes or probabilities found.")
    
except Exception as e:
    print(f"Error performing object detection: {e}")
```

## Acknowledgements
- Dr. Saleh Hussein, my supervisor for the time and space given to me to finish this work.

- Special thanks to Dr. Abdüssamet Aslan for providing the blood cell detection dataset. The original dataset can be found [here](https://github.com/draaslan/blood-cell-detection-dataset).

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.

---
