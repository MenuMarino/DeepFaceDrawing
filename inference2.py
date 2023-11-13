import torch
import os
import datasets, models, utils, metrics
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Inference')
    parser.add_argument('--weight', type=str, required=True, help='Path to load model weights.')
    parser.add_argument('--folder', type=str, default=None, help='Path to folder to be inference.')
    parser.add_argument('--output', type=str, required=True, help='Path to save result image.')
    parser.add_argument('--real_images', type=str, default=None, help='Path to folder of real images to compare.')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the model.')
    parser.add_argument('--generate', action='store_true', help='Generate additional images if set.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args

FILENAME = 'metrics_3_images.csv'

def add_noise(img):
    noise = np.random.normal(0, 0.1, img.size)
    noisy_img = np.array(img) + noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def calculate_metrics(generated_folder, real_folder):
    fid = metrics.calculate_fid(generated_folder, real_folder)
    is_score, _ = metrics.calculate_inception_score(generated_folder)
    # ms_ssim_score = metrics.calculate_average_ms_ssim(generated_folder, real_folder)
    return fid, is_score, 0

def inference(model, sketch_folder, generated_folder, real_folder, device, args):
    os.makedirs(generated_folder, exist_ok=True)
    
    image_files = os.listdir(sketch_folder)
    for file_name in tqdm(image_files, desc='Processing images', unit='image'):
        sketch_path = os.path.join(sketch_folder, file_name)
        generated_path = os.path.join(generated_folder, file_name)
        
        images = []
        original_image = datasets.dataloader.load_one_sketch(sketch_path, simplify=True, device=args.device).unsqueeze(0).to(device)
        images.append(original_image)
        
        if args.generate:
            pil_img = Image.open(sketch_path)
            mirror_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            noisy_img = add_noise(pil_img)

            images.append(datasets.dataloader.load_one_sketch(mirror_img, simplify=True, device=args.device).unsqueeze(0).to(device))
            images.append(datasets.dataloader.load_one_sketch(noisy_img, simplify=True, device=args.device).unsqueeze(0).to(device))

        with torch.no_grad():
            if len(images) == 1:
                result = model(images[0])
            else:
                result = model(images)
        result = utils.convert.tensor2PIL(result[0])
        result.save(generated_path)
        # print(f'Saved result to {generated_path}')
    
    fid, is_score, ms_ssim_score = calculate_metrics(generated_folder, real_folder)
    
    with open(FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([80, fid, is_score])
    print(f'Average Metrics - FID: {fid}, IS: {is_score}, MS-SSIM: {ms_ssim_score}')

def main(args):
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=False, IS2=True,
        manifold=args.manifold
    )
    model.load(args.weight)
    model.to(device)
    model.eval()
    
    if args.folder and args.real_images:
        if not os.path.exists(FILENAME):
            with open(FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epocas','FID (menos es mejor: 26.509)', 'IS (mayor es mejor: 2.317)'])
        
        inference(model, args.folder, args.output, args.real_images, device, args)
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)
