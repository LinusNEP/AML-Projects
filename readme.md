# Project (P5) Template: Open Vocabulary Object Detection (OVOD)

## Project Overview

Traditional detectors (like YOLO or Faster-RCNN) classify only objects they were trained on, a **closed vocabulary**. **OVOD** extends this by using **text embeddings** (from CLIP or similar models) that encode semantic meaning. Thus, the model can detect **new categories** via **cosine similarity** between image regions and text embeddings. In this project,  you will implement a complete OVOD pipeline using the [EnvoDat dataset](https://linusnep.github.io/EnvoDat/). You will:

1. **Train** a base detector on a subset of categories (seen classes)
2. **Evaluate** its ability to detect both seen and unseen objects
3. **Benchmark** performance against traditional closed-vocabulary detection
4. **Analyse** the generalisation capabilities for zero-shot detection

# Concepts

- **Seen Classes**: Objects the model is explicitly trained on
- **Unseen Classes**: Objects held out from training but present in the dataset for evaluation
- **Zero-shot Detection**: Detecting objects from categories not seen during training
- **Vision-Language Integration**: Using CLIP to match visual features with text embeddings

## Prerequisites

- Python ≥ 3.10 is installed on your system
- Basic knowledge of Python and command-line operations
- CUDA-enabled GPU (8 GB VRAM recommended)
- 20GB+ free disk space for datasets and models

## Environment Setup

1. Create a virtual environment with Python 3.8+ and install the necessary requirements. If Python 3 is not installed, download and install it from [Python's official site](https://www.python.org/downloads/).

   ```bash
   sudo apt-get install python3-venv
   python3 -m venv P5-OVOD
   source P5-OVOD/bin/activate
   ```
2. After the above step, you should see `(P5-OVOD)` at the start of your command prompt, indicating that the virtual environment is active. Once activated, install the necessary packages, e.g., ultralytics for YOLO and other relevant Python libraries:

   ```bash
   # A) Install PyTorch (pick CUDA build that matches the lab machines)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # B) Core libs and YOLO lib
   pip install numpy pandas matplotlib opencv-python pillow tqdm pyyaml pycocotools tensorboard rich ultralytics
   
   # C) Vision-language tooling
   pip install timm transformers sentencepiece
   pip install git+https://github.com/openai/CLIP.git
   ```

## Prepare the Dataset

1. Create a directory:

   ```bash
   mkdir -p ovod_ws/
   cd ovod_ws
   ```
2. Download only one of the following datasets in the directory:

   MU-Hall: <https://cloud.cps.unileoben.ac.at/index.php/s/yqaGDG2HiSqmEX4>

   MU-CPS: <https://cloud.cps.unileoben.ac.at/index.php/s/B7Dy7RgYDefqwsN>

   MU-TXN: <https://cloud.cps.unileoben.ac.at/index.php/s/jGJtn38rpxEXPWc>

   Leo-Str: <https://cloud.cps.unileoben.ac.at/index.php/s/SsK4xsBgGGinADL>

   Detailed information about the data can be found at [EnvoDat annotations](https://sites.google.com/view/envodat/download). We provide the data in different formats suitable for supervised model training, e.g., YOLOv*, COCO, OpenAI-CLIP, VOC, etc. We organised the dataset in the hierarchical structure shown in the following example with the COCO format:

```
data/mu-cps-coco/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
|    ├── images/
|    └── _annotations.coco.json
└── README.md
```

1. Rename the annotation JSON file:

   ```bash
   for s in train valid test; do
     mv data/mu-cps-coco/$s/_annotations.coco.json data/mu-cps-coco/$s/annotations.json
   done
   ```

The resulting annotations.json will contain the metadata in the following example:

```yaml
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 0, "name": "doors"},
    {"id": 1, "name": "box"},
    {"id": 2, "name": "broom"},
    ...
  ]
}
```

However, we want to end up with something similar to:

```bash
{
  "images": [...],
  "annotations": [...],
  "categories": [...],
  "ovod": {
    "seen": ["chair","table","person","monitor","door","window"],
    "unseen": ["robot","vacum cleaner","broom","soap dispenser","wash basin"]
  }
}
```

Pick a few common objects (that you want the model to learn directly) as **seen**, and hold out a few as **unseen** to test open-vocabulary generalisation. Example:

```bash
seen   = ["chair","table","person","monitor","door","window"]
unseen = ["robot","vacum cleaner","broom","soap dispenser","wash basin"]
```

The unseen classes **must exist** in your dataset’s `categories` list; otherwise, the evaluation code won’t find them. You can create a simple Python script in your project root to add the **seen** and **unseen** classes. Create and run the following `ovod_metadata.py:`

```python
import json, os, argparse, pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data/mu-cps-coco", help="Path to dataset root folder")
args = parser.parse_args()

# Define your seen/unseen split
seen = ["chair","table","person","monitor","door","window"]
unseen = ["robot","vacum cleaner","broom","soap dispenser","wash basin"]

root = pathlib.Path(args.root)
splits = ["train", "valid", "test"]

for split in splits:
    json_path = root / split / "annotations.json"

    if not json_path.exists():
        print(f"Missing file: {json_path}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    # Make sure classes exist
    category_names = {c["name"] for c in data["categories"]}
    missing_seen = set(seen) - category_names
    missing_unseen = set(unseen) - category_names

    if missing_seen or missing_unseen:
        print(f" Missing categories in {split}: {missing_seen | missing_unseen}")

    # Add ovod metadata
    data["ovod"] = {"seen": seen, "unseen": unseen}

    # Overwrite the JSON file (safe: you’re not touching images/annotations)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added ovod metadata to {json_path}")
```

After executing the above script in the directory, you will update the files:

```
data/mu-cps-coco/train/annotations.json
data/mu-cps-coco/valid/annotations.json
data/mu-cps-coco/test/annotations.json
```

You can verify if the ovod data is added by opening any of the json files and scrolling to the bottom, or programmatically:

```bash
import json
data = json.load(open("data/mu-cps-coco/train/annotations.json"))
print("Seen classes:", data["ovod"]["seen"])
print("Unseen classes:", data["ovod"]["unseen"])
print("Total categories:", len(data["categories"]))
```

# Train the base Detector Model on the Seen classes

For this, follow the instructions provided at the EnvoDat repository:

<https://github.com/LinusNEP/EnvoDat/blob/main/docs/GET_STARTED.md>

## **OVOD Inference Pipeline**

1. Create `src/ovod_detector.py`:

```python
import torch
import clip
from ultralytics import YOLO
import cv2
import json

class OVODDetector:
    def __init__(self, detector_weights, dataset_path, clip_model="ViT-B/32"):
        # Load base detector (trained on seen classes only)
        self.detector = YOLO(detector_weights)
        
        # Load CLIP for open-vocabulary classification
        self.clip_model, self.clip_preprocess = clip.load(clip_model)
        self.clip_model.eval()
        
        # Load class splits from dataset
        with open(f'{dataset_path}/valid/annotations.json', 'r') as f:
            data = json.load(f)
        
        self.seen_classes = data['ovod']['seen']
        self.unseen_classes = data['ovod']['unseen']
        
        print(f"Loaded: {len(self.seen_classes)} seen, {len(self.unseen_classes)} unseen classes")
    
    def encode_text_prompts(self, class_names):
        """Encode class names to CLIP text features"""
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {cls}") for cls in class_names])
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def predict_ovod(self, image_path, confidence_thresh=0.3):
        """Run OVOD inference combining base detector and CLIP"""
        # Get base detections from seen classes
        base_results = self.detector(image_path)
        all_detections = []
        
        # Process base class detections
        for result in base_results:
            if result.boxes is not None:
                for box in result.boxes:
                    if box.conf.item() > confidence_thresh:
                        all_detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': box.conf.item(),
                            'class_id': int(box.cls.item()),
                            'class_name': self.seen_classes[int(box.cls.item())],
                            'type': 'seen'
                        })
        
        # TODO: Add CLIP-based detection for unseen classes
        # This would involve region proposals + CLIP classification
        # for the predefined unseen classes
        
        return all_detections

# Usage example
if __name__ == "__main__":
    detector = OVODDetector(
        detector_weights='models/base_detector_seen_classes/weights/best.pt',
        dataset_path='data/mu-cps-coco'
    )
    
    results = detector.predict_ovod('path/to/test_image.jpg')
    for det in results:
        print(f"{det['class_name']} ({det['type']}): {det['confidence']:.2f}")
```

1. Run inference:

   ```python
   python src/ovod_detector.py
   ```

## **Evaluation and Benchmarking**

1. Create `src/evaluate_ovod.py`:

   ```python
   import json
   from pycocotools.coco import COCO
   from pycocotools.cocoeval import COCOeval
   
   def evaluate_ovod_performance(detector, dataset_path, output_dir):
       """Evaluate OVOD performance on seen and unseen classes"""
       
       # Load dataset metadata
       with open(f'{dataset_path}/valid/annotations.json', 'r') as f:
           data = json.load(f)
       
       seen_classes = data['ovod']['seen']
       unseen_classes = data['ovod']['unseen']
       
       # Load ground truth
       coco_gt = COCO(f'{dataset_path}/valid/annotations.json')
       
       # Run inference and collect results
       # ... implementation details ...
       
       # Calculate metrics
       results = {
           'seen_mAP50': calculate_seen_map(detector, coco_gt, seen_classes),
           'unseen_mAP50': calculate_unseen_map(detector, coco_gt, unseen_classes),
           'harmonic_mean': calculate_harmonic_mean(seen_mAP50, unseen_mAP50)
       }
       
       print("OVOD Evaluation Results:")
       print(f"Seen Classes mAP50: {results['seen_mAP50']:.3f}")
       print(f"Unseen Classes mAP50: {results['unseen_mAP50']:.3f}") 
       print(f"Harmonic Mean: {results['harmonic_mean']:.3f}")
       
       return results
   
   if __name__ == "__main__":
       from ovod_detector import OVODDetector
       
       detector = OVODDetector(
           'models/base_detector_seen_classes/weights/best.pt',
           'data/mu-cps-coco'
       )
       
       results = evaluate_ovod_performance(
           detector, 
           'data/mu-cps-coco',
           'results/'
       )
   ```
2. Run evaluation

   ```python
   python src/evaluate_ovod.py
   ```

## **Results Analysis**

Create visualisations and tables for your final report. Example:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ovod_comparison(results):
    """Plot comparative results for OVOD evaluation"""
    metrics = ['Seen mAP50', 'Unseen mAP50', 'Harmonic Mean']
    values = [results['seen_mAP50'], results['unseen_mAP50'], results['harmonic_mean']]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=values, palette=['blue', 'red', 'green'])
    plt.title('OVOD Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/ovod_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## **Project Structure**

1. The final project files to be submitted must be organised as follows:

```bash
ovod_ws/
├── data/
│   └── mu-cps-coco/          # Dataset with OVOD metadata
├── models/                   # Trained model weights
├── src/                      # Source code
│   ├── ovod_detector.py      # Main OVOD detector
│   ├── evaluate_ovod.py      # Evaluation scripts
│   ├── visualize_results.py  # Visualization utilities
│   └── utils/                # Helper functions
├── configs/                  # Training configurations
├── results/                  # Evaluation results and plots
└── README.md
```

1. Your final report should be in pdf using the project submission template provided during the lecture. You can use Overleaf for your writing:[  https://www.overleaf.com/](https://www.overleaf.com/edu/leoben#templates)
