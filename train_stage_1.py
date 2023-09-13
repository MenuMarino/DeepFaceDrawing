import numpy as np
import torch
import pickle
from tqdm import tqdm
from scipy.spatial import cKDTree
import datasets, models, losses

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 1')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comet', type=str, default=None, help='comet.ml API')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping')
    parser.add_argument('--comet_log_image', type=str, default=None, help='Path to model input image to be inference and log the result to comet.ml. Skipped if --comet is not given.')
    args = parser.parse_args()
    return args

def validation_parser(args):
    if not args.comet:
        if args.comet_log_image: print('args.comet_log_image will be skipped.')
    

def save_points(all_points):
    combined_points = np.array(all_points)
    tree = cKDTree(combined_points)
    with open('./tree.pickle', 'wb') as f:
        pickle.dump(tree, f)

def main(args):
    best_loss = float('inf')
    epochs_no_improve = 0
    device = torch.device(args.device)
    all_points = []
    print(f'Device : {device}')
    
    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=True,
        FM=False, FM_decoder=False,
        IS=False, IS_generator=False, IS_discriminator=False, IS2=False,
        manifold=False
    )
    
    if args.resume:
        model.load(args.resume, map_location=device)
        
    model.to(device)

    train_dataloader = datasets.dataloader.dataloader(args.dataset, batch_size=args.batch_size, load_photo=False, augmentation=False)
    
    if args.dataset_validation:
        validation_dataloader = datasets.dataloader.dataloader(args.dataset_validation, batch_size=args.batch_size, load_photo=False)

    optimizer = torch.optim.Adam(model.CE.parameters(), lr=0.0002, betas=(0.5, 0.999))
    mse = losses.MSE()
        
    for epoch in range(args.epochs):
        all_points = []
        running_loss = {
            'loss_left_eye' : 0,
            'loss_right_eye' : 0,
            'loss_nose' : 0,
            'loss_mouth' : 0,
            'loss_background' : 0
        }
        
        model.train()
        for sketches in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            iteration_loss = {
                'loss_left_eye_it' : 0,
                'loss_right_eye_it' : 0,
                'loss_nose_it' : 0,
                'loss_mouth_it' : 0,
                'loss_background_it' : 0
            }
            
            optimizer.zero_grad()

            sketches = sketches.to(device)
            patches = model.CE.crop(sketches)
            points = model.CE.encode(patches)
            all_points.extend(points['left_eye'].cpu().detach().numpy().tolist())
            all_points.extend(points['right_eye'].cpu().detach().numpy())
            all_points.extend(points['nose'].cpu().detach().numpy())
            all_points.extend(points['mouth'].cpu().detach().numpy())
            all_points.extend(points['background'].cpu().detach().numpy())
            repatches = model.CE.decode(points)
            
            for key in model.components:
                loss = mse.compute(repatches[key], patches[key])
                loss.backward()
                iteration_loss[f'loss_{key}_it'] += loss.item()
            
            optimizer.step()
                
            for key, loss in iteration_loss.items():
                running_loss[key[:-3]] += loss * len(sketches) / len(train_dataloader.dataset)
        
        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_left_eye' : 0,
                'val_loss_right_eye' : 0,
                'val_loss_nose' : 0,
                'val_loss_mouth' : 0,
                'val_loss_background' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):
                    validation_iteration_loss = {
                        'val_loss_left_eye_it' : 0,
                        'val_loss_right_eye_it' : 0,
                        'val_loss_nose_it' : 0,
                        'val_loss_mouth_it' : 0,
                        'val_loss_background_it' : 0
                    }

                    sketches = sketches.to(device)
                    patches = model.CE.crop(sketches)
                    repatches = model.CE.decode(model.CE.encode(patches))

                    for key in model.components:
                        loss = mse.compute(repatches[key], patches[key])
                        validation_iteration_loss[f'val_loss_{key}_it'] += loss.item()
                        
                    for key, loss in validation_iteration_loss.items():
                        validation_running_loss[key[:-3]] += loss * len(sketches) / len(validation_dataloader.dataset)

        
        def print_dict_loss(dict_loss):
            for key, loss in dict_loss.items():
                print(f'Loss {key:12} : {loss:.6f}')
                
        print()    
        print(f'Epoch - {epoch+1} / {args.epochs}')
        print_dict_loss(running_loss)
        if args.dataset_validation: print_dict_loss(validation_running_loss)
        print()
        
        if args.output:
            model.save(args.output)

        avg_val_loss = sum(validation_running_loss.values()) / len(validation_running_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == args.patience:
                print("Early stopping!")
                break
            
    save_points(all_points)

if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    validation_parser(args)
    main(args)