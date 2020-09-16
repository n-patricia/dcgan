import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from hmc import get_samples

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

img_list = []
G_losses = []
D_losses = []
iters = 0

def dcgan(data_loader, netG, netD, args):
    device = args.device
    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    paramsG = [p for p in netG.parameters() if p.requires_grad]
    paramsD = [p for p in netD.parameters() if p.requires_grad]
    optimizerG = optim.Adam(paramsG, lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(paramsD, lr=args.lr, betas=(args.beta1, 0.999))
    writer = SummaryWriter('./runs/{}_exp_1'.format(args.dataset))

    dataiter = iter(data_loader)
    img, _ = dataiter.next()
    writer.add_graph(netG, make_grid(img.detach()))

    print('Starting training ...')
    # For each epoch
    for epoch in tqdm(range(args.num_epochs)):
        # For each batch in the data_loader
        for i, data in enumerate(data_loader, 0):
            ###########################
            # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float32)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.rand(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            #############################
            # (2) Update G network; maximize log(D(G(z)))
            #############################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward(retain_graph=True)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i%50==0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, args.num_epochs, i, len(data_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            
            if (iters%500==0) or ((epoch==args.num_epochs-1) and (i==len(data_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach()
                fake_grid = make_grid(fake, padding=2)
                writer.add_image('fake_image_epoch_{}'.format(epoch))
                writer.add_scalar('loss/generator', errG.item(), iters)
                writer.add_scalar('loss/discriminator', errD.item(), iters)
                writer.add_scalar('disc_output/z1', D_G_z1, iters)
                writer.add_scalar('disc_output/z2', D_G_z2, iters)

            iters += 1
            
        print('End of epoch: {}\n'.format(epoch))

        if epoch%10==0:
            torch.save(netG.state_dict(), os.path.join('{}/{}'.format(args.checkpoint_path, args.dataset), 'netG_dcgan_epoch_{}.pth'.format(epoch)))



def presgan(data_loader, netG, netD, log_sigma, args):
    device = args.device
    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    paramsG = [p for p in netG.parameters() if p.requires_grad]
    paramsD = [p for p in netD.parameters() if p.requires_grad]
    optimizerG = optim.Adam(paramsG, lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(paramsD, lr=args.lr, betas=(args.beta1, 0.999))
    optimizerSigma = optim.Adam([log_sigma], lr=args.sigma_lr, betas=(args.beta1, 0.999))

    writer = SummaryWriter('./runs/{}_exp_1'.format(args.dataset))
    stepsize = args.stepsize_num / args.nz

    dataiter = iter(data_loader)
    img, _ = dataiter.next()
    writer.add_graph(netG, make_grid(img.detach()))

    print('Starting training ...')
    # For each epoch
    for epoch in tqdm(range(args.num_epochs)):
        # For each batch in the data_loader
        for i, data in enumerate(data_loader, 0):
            sigma_x = F.softplus(log_sigma).view(1, 1, args.image_size, args.image_size)

            ## Update D network
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float32)
            
            # create noise based on real data
            noise_eta = torch.randn_like(real_cpu)
            noised_data = real_cpu + sigma_x.detach() * noise_eta
            output = netD(noised_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            mu_fake = netG(noise)
            fake = mu_fake + sigma_x * noise_eta
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ## Update G network
            netG.zero_grad()
            optimizerSigma.zero_grad()
            
            label.fill_(real_label)
            gen_input = torch.randn(b_size, args.nz, 1, 1, device=device)
            output = netG(gen_input)
            noise_eta = torch.randn_like(output)
            fake = output + noise_eta * sigma_x

            fake_dec = netD(fake)
            errG_gan = criterion(fake_dec, label)
            D_G_z2 = fake_dec.mean().item()
            
            hmc_samples, acceptRate, stepsize = get_samples(netG, fake, gen_input.clone(), sigma_x.detach(), args.burn_in, args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt, args.hmc_learning_rate, args.hmc_opt_accept)
            b_size, d = hmc_samples.size()
            mean_output = netG(hmc_samples.view(b_size, d, 1, 1).to(device))
            b_size = fake.size(0)

            mean_output_summed = torch.zeros_like(fake)
            for cnt in range(args.num_samples_posterior):
                mean_output_summed = mean_output_summed + mean_output[cnt*b_size:(cnt+1)*b_size]
            mean_output_summed = mean_output_summed / args.num_samples_posterior

            c = ((fake - mean_output_summed)/sigma_x**2).detach()
            errG_entropy = torch.mul(c, output + sigma_x * noise_eta).mean(0).sum()
            errG = errG_gan - args.lambda_ * errG_entropy
            errG.backward(retain_graph=True)
            optimizerG.step()
            optimizerSigma.step()

            if i%50==0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(data_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            if (iters%500==0) or ((epoch==args.num_epochs-1) and (i==len(data_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach()
                fake_grid = make_grid(fake, padding=2)
                writer.add_image('fake_image_epoch_{}'.format(epoch))
                writer.add_scalar('loss/generator', errG.item(), iters)
                writer.add_scalar('loss/discriminator', errD.item(), iters)
                writer.add_scalar('disc_output/z1', D_G_z1, iters)
                writer.add_scalar('disc_output/z2', D_G_z2, iters)

            iters += 1

        print('sigma min: {} - max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
        print('MCMC diagnostics =====> stepsize: {} | min ar: {} | max ar: {} | mean ar: {}'.format(stepsize, acceptRate.min().item(), acceptRate.max().item(), acceptRate.mean().item()))
        print('End of epoch: {}\n'.format(epoch))

        if epoch%10==0:
            torch.save(netG.state_dict(), os.path.join('{}/{}'.format(args.checkpoint_path, args.dataset), 'netG_presgan_epoch_{}.pth'.format(epoch)))
            torch.save(log_sigma, os.path.join('{}/{}'.format(args.checkpoint_path, args.dataset), 'log_sigma_epoch_{}.pth'.format(epoch)))
