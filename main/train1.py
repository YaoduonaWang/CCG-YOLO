from PIL.ImageOps import scale

from ultralytics import YOLO
from pathlib import Path

# Load a model
model_ms = YOLO("/root/awsyolo/ultralytics/cfg/models/11/c3k2.yaml") # build a new model from YAML
run_name = "Rhododendron_Lundy_V2_NIR_RE_312py"

dataset_root = Path("/root/awsyolo/")  # 
dataset_name = "Rhododendron_Lundy_V2_NIR_RE"
#/Users/diona/Desktop/Lundy_Rhododendron_dataset_V2/Rhododendron_Lundy_V2_NIR_RE
dataset_path = dataset_root / dataset_name / f"{dataset_name}.yaml"
dataset_path = str(dataset_path)

# Train the model
if __name__ == '__main__':

    results = model_ms.train(data=dataset_path, # Line to run multispectral training
        epochs=300,
        imgsz=640,
        batch = 6,
        device = "1",
        name = run_name,
        optimizer = "AdamW",
        #augment = False,
        amp = False,
        patience = 200,
        lr0=0.01,  # Starting learning rate
        seed = 0
    )