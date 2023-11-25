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
    --epochs 15 \
    --resume weight3/weight/DeepFaceDrawing/ \
    --output weight3/weight/DeepFaceDrawing/ \
    --device cuda
```

weight tiene 100
weight2 tiene 80
weight3 120

## CLI Inference

**Single Image Inference**

The following commands will inference `images.jpg` and then saved the result to `output.jpg`.

```
python inference.py \
    --weight weight/weight/DeepFaceDrawing/ \
    --image testing/20.jpg \
    --output testing/20-generated-test.jpg \
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

```
python inference2.py \
    --weight weight3/weight/DeepFaceDrawing/ \
    --folder person-face-sketches/test/sketches/ \
    --output person-face-sketches/test/generated-w3-images-todo/ \
    --real_images person-face-sketches/test/photos/ \
    --device cuda \
    --generate
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
