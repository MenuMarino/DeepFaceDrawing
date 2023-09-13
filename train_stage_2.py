import torch
import datasets, models, losses
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="UserWarning")

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage 2')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--resume_CE', type=str, default=None, help='Path to load Component Embedding model weights. Required if --resume is not given. Skipped if --resume is given.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comet', type=str, default=None, help='comet.ml API')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping')
    parser.add_argument('--comet_log_image', type=str, default=None, help='Path to model input image to be inference and log the result to comet.ml. Skipped if --comet is not given.')
    args = parser.parse_args()
    return args

def validation_parser(args):
    if args.resume:
        if args.resume_CE: print('args.resume_CE will be skipped.')
    else:
        assert args.resume_CE, "Both args.resume and args.resume_CE can't be None." 
    if not args.comet:
        if args.comet_log_image: print('args.comet_log_image will be skipped.')
    
def main(args):
    
    device = torch.device(args.device)
    print(f'Device : {device}')

    model = models.DeepFaceDrawing(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=True, IS2=True,
        manifold=False
    )

    if args.resume:
        model.load(args.resume, map_location=device)
    else:
        model.CE.load(args.resume_CE, map_location=device)
    
    model.to(device)
    
    train_dataloader = datasets.dataloader.dataloader(args.dataset, batch_size=args.batch_size, load_photo=True, augmentation=False)
    
    if args.dataset_validation:
        validation_dataloader = datasets.dataloader.dataloader(args.dataset_validation, batch_size=args.batch_size, load_photo=True)
    
    for key, component in model.CE.components.items():
        for param in component.parameters():
            param.requires_grad = False

    optimizer_generator = torch.optim.Adam( list(model.FM.parameters()) + list(model.IS.G.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam( list(model.IS.D1.parameters()) + list(model.IS.D2.parameters()) + list(model.IS.D3.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator2 = torch.optim.Adam( list(model.FM.parameters()) + list(model.IS2.G.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator2 = torch.optim.Adam( list(model.IS2.D1.parameters()) + list(model.IS2.D2.parameters()) + list(model.IS2.D3.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    
    l1 = losses.L1()
    bce = losses.BCE()
    perceptual = losses.Perceptual(device=args.device)

    label_real = model.IS.label_real
    label_fake = model.IS.label_fake
    label_real2 = model.IS2.label_real
    label_fake2 = model.IS2.label_fake
    
    for epoch in range(args.epochs):
        
        running_loss = {
            'loss_G' : 0,
            'loss_D' : 0,
            'loss_G2' : 0,
            'loss_D2' : 0
        }
        
        model.train()
        for sketches, photos in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):
            
            sketches = sketches.to(device)
            photos = photos.to(device)
            
            latents = model.CE.encode(model.CE.crop(sketches))
            spatial_map = model.FM.merge(model.FM.decode(latents))
            fake_photos = model.IS.generate(spatial_map)
            
            optimizer_generator.zero_grad()
            loss_G_L1 = l1.compute(fake_photos, photos)
            loss_perceptual = perceptual.compute(fake_photos, photos)
            patches = model.IS.discriminate(spatial_map, fake_photos)
            loss_G_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE
            # loss_G.backward(retain_graph=True)
            optimizer_generator.step()
            
            optimizer_discriminator.zero_grad()
            patches = model.IS.discriminate(spatial_map.detach(), fake_photos.detach())
            loss_D_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            patches = model.IS.discriminate(spatial_map.detach(), photos.detach())
            loss_D_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_D = loss_D_fake + loss_D_real
            # loss_D.backward(retain_graph=True)
            optimizer_discriminator.step()
            
            # Nuevo
            fake_photos2 = model.IS2.generate(fake_photos)
            optimizer_generator2.zero_grad()
            loss_G2_L1 = l1.compute(fake_photos2, photos)
            loss_perceptual = perceptual.compute(fake_photos2, photos)
            patches = model.IS2.discriminate(fake_photos, fake_photos2)
            loss_G2_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real2, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_G2 = loss_perceptual + 10 * loss_G2_L1 + loss_G2_BCE
            # loss_G2.backward(retain_graph=True)
            optimizer_generator2.step()
            
            optimizer_discriminator2.zero_grad()
            patches = model.IS2.discriminate(fake_photos.detach(), fake_photos2.detach())
            loss_D2_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake2, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            patches = model.IS2.discriminate(fake_photos.detach(), photos.detach())
            loss_D2_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real2, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_D2 = loss_D2_fake + loss_D2_real
            # loss_D2.backward(retain_graph=True)
            optimizer_discriminator2.step()
            (loss_G + loss_G2).backward()
            (loss_D + loss_D2).backward()

            iteration_loss = {
                'loss_G_it' : loss_G.item(),
                'loss_D_it' : loss_D.item(),
                'loss_G2_it' : loss_G2.item(),
                'loss_D2_it' : loss_D2.item()
            }

            for key, loss in iteration_loss.items():
                running_loss[key[:-3]] = loss * len(sketches) / len(train_dataloader.dataset)
        
        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_G' : 0,
                'val_loss_D' : 0,
                'val_loss_G2' : 0,
                'val_loss_D2' : 0
            }
            
            model.eval()
            with torch.no_grad():
                for sketches, photos in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):
            
                    sketches = sketches.to(device)
                    photos = photos.to(device)
                    
                    latents = model.CE.encode(model.CE.crop(sketches))
                    spatial_map = model.FM.merge(model.FM.decode(latents))
                    fake_photos = model.IS.generate(spatial_map)
                    
                    loss_G_L1 = l1.compute(fake_photos, photos)
                    loss_perceptual = perceptual.compute(fake_photos, photos)
                    patches = model.IS.discriminate(spatial_map, fake_photos)
                    loss_G_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE
                    
                    patches = model.IS.discriminate(spatial_map.detach(), fake_photos.detach())
                    loss_D_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    patches = model.IS.discriminate(spatial_map.detach(), photos.detach())
                    loss_D_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    loss_D = loss_D_fake + loss_D_real
                    
                    # Segundo GAN
                    fake_photos2 = model.IS2.generate(fake_photos)

                    loss_G2_L1 = l1.compute(fake_photos2, photos)
                    loss_perceptual = perceptual.compute(fake_photos2, photos)
                    patches = model.IS2.discriminate(fake_photos, fake_photos2)
                    loss_G2_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real2, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    loss_G2 = loss_perceptual + 10 * loss_G2_L1 + loss_G2_BCE

                    patches = model.IS2.discriminate(fake_photos.detach(), fake_photos2.detach())
                    loss_D2_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake2, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    patches = model.IS2.discriminate(fake_photos.detach(), photos.detach())
                    loss_D2_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real2, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    loss_D2 = loss_D2_fake + loss_D2_real

                    validation_iteration_loss = {
                        'val_loss_G_it' : loss_G.item(),
                        'val_loss_D_it' : loss_D.item(),
                        'val_loss_G2_it' : loss_G2.item(),
                        'val_loss_D2_it' : loss_D2.item()
                    }

                    for key, loss in validation_iteration_loss.items():
                        validation_running_loss[key[:-3]] = loss * len(sketches) / len(validation_dataloader.dataset)
                        
            avg_val_loss = sum(validation_running_loss.values()) / len(validation_running_loss)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == args.patience:
                    print("Early stopping!")
                    break
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
        
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    validation_parser(args)
    main(args)