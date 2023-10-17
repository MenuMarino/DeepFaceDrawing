## Env

`source ~/tesis/Scripts/activate`

## Training

**Stage 1: Component Embedding**

```
python train_stage_1.py \
    --dataset person-face-sketches/train/ \
    --dataset_validation person-face-sketches/val/ \
    --batch_size 2 \
    --epochs 12 \
    --output weight/weight/DeepFaceDrawing/ \
    --device cuda \
    --resume weight/weight/DeepFaceDrawing/
```

**Stage 2: Feature Mapping and Image Synthesis**

```
python train_stage_2.py \
    --dataset person-face-sketches/train/ \
    --dataset_validation person-face-sketches/val/ \
    --batch_size 2 \
    --epochs 17 \
    --resume weight/weight/DeepFaceDrawing/ \
    --output weight/weight/DeepFaceDrawing/ \
    --device cuda
```

Epocas: 60

## CLI Inference

**Single Image Inference**

The following commands will inference `images.jpg` and then saved the result to `output.jpg`.

```
python inference.py \
    --weight weight/weight/DeepFaceDrawing/ \
    --image testing/20.jpg \
    --output testing/20-real-7.jpg \
    --device cuda \
    --manifold
```

```
python inference.py \
    --weight weight/weight/DeepFaceDrawing/ \
    --folder testing_folder/ \
    --output testing_output_folder/ \
    --device cuda \
    --manifold
```

**Folder of Images Inference**

The following commands will inference all images in the folder `input_folder` and then saved the result to `output_folder`.

```
python inference.py \
    --weight weight/weight/DeepFaceDrawing/ \
    --folder input_folder/ \
    --output output_folder/ \
    --device cuda
```

## Web Application

The following commands will host the model as web-version application at `localhost:8000`.

```
python web.py \
    --weight weight/weight/DeepFaceDrawing/ \
    --device cuda \
    --host 0.0.0.0 \
    --port 8000
```
