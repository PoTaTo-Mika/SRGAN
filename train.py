import argparse
import os
from math import log10

import logging
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from models.model import Generator, Discriminator

from datetime import datetime

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # --- Logging Setup ---
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_srf_{UPSCALE_FACTOR}_{timestamp}.log')

    # Get the root logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO) # Set the minimum logging level

    # Create handlers
    # File handler to write logs to a file
    fh = logging.FileHandler(log_filename, mode='w') # Use 'w' to overwrite for each new run, 'a' to append
    fh.setLevel(logging.INFO)

    # Console handler to output logs to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # You can set this to DEBUG for more verbose console output

    # Create formatter and add it to the handlers
    # Define the log message format: timestamp - logger name - level - message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    # This check prevents adding handlers multiple times if the script is run interactively
    if not logger.handlers:
         logger.addHandler(fh)
         logger.addHandler(ch)

    # Use logging.info instead of print for initial messages
    logging.info("Starting training process...")
    logging.info(f"Parameters: Crop Size={CROP_SIZE}, Upscale Factor={UPSCALE_FACTOR}, Epochs={NUM_EPOCHS}")
    # --- End Logging Setup ---


    train_set = TrainDatasetFromFolder('../data/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('../data/val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    # Use logging.info instead of print
    logging.info(f'# generator parameters: {sum(param.numel() for param in netG.parameters())}')
    netD = Discriminator()
    # Use logging.info instead of print
    logging.info(f'# discriminator parameters: {sum(param.numel() for param in netD.parameters())}')

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        logging.info("Using CUDA for training.")
    else:
        logging.info("Using CPU for training.")


    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # Log the start of the epoch
        logging.info(f'--- Epoch {epoch}/{NUM_EPOCHS} ---')

        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            # g_update_first = True # This variable is not used
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data # LR image
            if torch.cuda.is_available():
                z = z.float().cuda()

            # Train D
            optimizerD.zero_grad()
            real_out = netD(real_img).mean()
            # Detach fake images when training D
            fake_img_detached = netG(z).detach()
            fake_out_d = netD(fake_img_detached).mean() # Output from D for detached fake images
            d_loss = 1 - real_out + fake_out_d

            d_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ##########################
            optimizerG.zero_grad()
            fake_img = netG(z) # Generate fake images again for G training
            fake_out_g = netD(fake_img).mean() # Output from D for fake images (not detached)

            # Use the generated fake_img and real_img for generator loss
            g_loss = generator_criterion(fake_out_g, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()


            # Update running results *after* the optimization steps
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size # D(x)
            running_results['g_score'] += fake_out_g.item() * batch_size # D(G(z)) for G's objective


            # tqdm set_description updates the console progress bar
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # --- Log End-of-Epoch Training Results ---
        epoch_d_loss = running_results['d_loss'] / running_results['batch_sizes']
        epoch_g_loss = running_results['g_loss'] / running_results['batch_sizes']
        epoch_d_score = running_results['d_score'] / running_results['batch_sizes']
        epoch_g_score = running_results['g_score'] / running_results['batch_sizes']
        # Use logging.info to record this summary
        logging.info(f"Epoch {epoch} Training Summary: Loss_D={epoch_d_loss:.4f}, Loss_G={epoch_g_loss:.4f}, D(x)={epoch_d_score:.4f}, D(G(z))={epoch_g_score:.4f}")
        # --- End Log Training Results ---


        netG.eval()
        out_path = 'training_results/SRGAN_ResNet_IN_FCA_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            # Reset validation results for the current epoch
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = [] # List to store validation images for saving
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0) # In val_loader, batch_size is 1
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()

                sr = netG(lr) # Generate super-resolved image

                # Calculate metrics for the current batch (which is size 1)
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size # Accumulate MSE
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size # Accumulate SSIM

                # Calculate average PSNR and SSIM over the validation set processed so far
                avg_mse = valing_results['mse'] / valing_results['batch_sizes']
                # Calculate PSNR, handle potential division by zero if MSE is extremely small
                avg_psnr = 10 * log10((hr.max().item()**2) / avg_mse) if avg_mse > 1e-10 else float('inf') # Use .item() to get scalar
                avg_ssim = valing_results['ssims'] / valing_results['batch_sizes']

                # Update the current average metrics in valing_results for display
                valing_results['psnr'] = avg_psnr
                valing_results['ssim'] = avg_ssim

                # tqdm set_description updates the console progress bar
                val_bar.set_description(
                    desc='[Validating] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                # Collect images for saving. Limit the number of images saved per epoch to avoid excessive files.
                # Collect LR, HR_restore (bicubic), and SR images for visual comparison.
                if len(val_images) < 15 * 3: # Example: save 15 sets of images (LR, HR_restore, SR)
                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), # Original LR upsampled (bicubic)
                         display_transform()(hr.data.cpu().squeeze(0)),     # High Resolution ground truth
                         display_transform()(sr.data.cpu().squeeze(0))])    # Super-Resolved output


            # --- Log End-of-Epoch Validation Results ---
            # After the loop, valing_results['psnr'] and valing_results['ssim'] hold the final averages for the epoch
            final_psnr = valing_results['psnr']
            final_ssim = valing_results['ssim']
            # Use logging.info to record this summary
            logging.info(f"Epoch {epoch} Validation Summary: PSNR={final_psnr:.4f} dB, SSIM={final_ssim:.4f}")
            # --- End Log Validation Results ---


            # Save validation images
            if len(val_images) > 0:
                # Stack the collected images and chunk them (each chunk will be a set of LR_restore, HR, SR)
                val_images = torch.stack(val_images)
                # Chunk into groups of 3 (LR_restore, HR, SR)
                val_images_chunked = torch.chunk(val_images, len(val_images) // 3)
                val_save_bar = tqdm(val_images_chunked, desc='[saving validation images]')
                index = 1
                for image_chunk in val_save_bar:
                    # Arrange images in a grid, nrow=3 means 3 images per row (LR_restore, HR, SR)
                    image = utils.make_grid(image_chunk, nrow=3, padding=5)
                    utils.save_image(image, os.path.join(out_path, 'epoch_%d_index_%d.png' % (epoch, index)), padding=5)
                    index += 1
                logging.info(f"Saved {len(val_images_chunked)} validation image sets for epoch {epoch}")


        # save model parameters
        model_save_dir = 'epochs/SRGAN_ResNet_IN_FCA_'
        g_dir = os.path.join(model_save_dir, 'G')
        d_dir = os.path.join(model_save_dir, 'D')
        
        os.makedirs(g_dir, exist_ok=True)
        os.makedirs(d_dir, exist_ok=True)
        
        torch.save(netG.state_dict(), os.path.join(g_dir, 'netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
        torch.save(netD.state_dict(), os.path.join(d_dir, 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
        logging.info(f"Saved models for epoch {epoch}")

        # save loss\scores\psnr\ssim to results dictionary
        results['d_loss'].append(epoch_d_loss) # Append the epoch average losses/scores
        results['g_loss'].append(epoch_g_loss)
        results['d_score'].append(epoch_d_score)
        results['g_score'].append(epoch_g_score)
        results['psnr'].append(final_psnr)       # Append the final epoch average PSNR
        results['ssim'].append(final_ssim)       # Append the final epoch average SSIM

        # Save results to CSV periodically and at the end
        stats_save_dir = 'statistics'
        if not os.path.exists(stats_save_dir):
            os.makedirs(stats_save_dir)
        if epoch % 10 == 0 or epoch == NUM_EPOCHS: # Save every 10 epochs or on the last epoch
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            csv_filename = os.path.join(stats_save_dir, 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv')
            data_frame.to_csv(csv_filename, index_label='Epoch')
            logging.info(f"Saved statistics to {csv_filename} up to epoch {epoch}")

    # Log the end of the training process
    logging.info("Training finished.")