import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.metrics import structural_similarity as ssim
import scipy
from PIL import Image

COMPONENTS = {
    'left_eye': (int(108), int(126), int(128)),
    'right_eye': (int(255), int(126), int(128)),
    'nose': (int(182), int(232), int(160)),
    'mouth': (int(169), int(301), int(192)),
    'background': (0, 0, int(512))
}

def load_and_preprocess_image(image_path, component = None):
    image = Image.open(image_path)
    if component and component in COMPONENTS:
        x, y, size = COMPONENTS[component]
        crop_area = (x, y, x + size, y + size)
        image = image.crop(crop_area)
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return preprocess_input(image)

def calculate_fid(generated_folder, real_folder, component):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    generated_images = [load_and_preprocess_image(os.path.join(generated_folder, img), component) for img in os.listdir(generated_folder)]
    real_images = [load_and_preprocess_image(os.path.join(real_folder, img), component) for img in os.listdir(real_folder)]
    
    gen_features = inception_model.predict(np.array(generated_images))
    real_features = inception_model.predict(np.array(real_images))
    
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    
    sum_sqrd_diff = np.sum((mu_gen - mu_real)**2)
    cov_mean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
    
    if not np.isfinite(cov_mean).all():
        msg = f'Product of covariances is not finite! sqrtm result: {cov_mean}'
        print(msg)
        return 1e10
    
    fid = sum_sqrd_diff + np.trace(sigma_gen + sigma_real - 2*cov_mean)
    return fid

def calculate_inception_score(generated_folder, component):
    inception_model = InceptionV3(include_top=True, input_shape=(299, 299, 3))
    generated_images = [load_and_preprocess_image(os.path.join(generated_folder, img), component) for img in os.listdir(generated_folder)]
    preds = inception_model.predict(np.array(generated_images))
    scores = []
    for part in np.array_split(preds, 10):
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def calculate_average_ms_ssim(generated_folder, real_folder):
    scores = []
    for img_gen_name, img_real_name in zip(os.listdir(generated_folder), os.listdir(real_folder)):
        img_gen_path = os.path.join(generated_folder, img_gen_name)
        img_real_path = os.path.join(real_folder, img_real_name)
        
        img_gen = load_and_preprocess_image(img_gen_path)
        img_real = load_and_preprocess_image(img_real_path)
        
        scores.append(ssim(img_gen, img_real, multichannel=True))
        
    return np.mean(scores)

