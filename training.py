import torch
import torch.optim as optim
import os
import random
import time
from utils.utils import *
from utils.mss_loss import multi_scale_spectrogram_loss
from models.diffusionmodel import DiffusionModel
from utils.plotters import *

def train(params, signals_list):
    if params.manual_random_seed != -1:
        random.seed(params.manual_random_seed)
        torch.manual_seed(params.manual_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    fs_list = params.fs_list
    n_scales = len(params.scales)
    diffusion_models = []
    noise_amp_list = []
    energy_list = [(sig ** 2).mean().item() for sig in signals_list]
    reconstruction_noise_list = []
    output_signals = []
    loss_vectors = []

    for scale_idx in range(n_scales):
        output_signals_single_scale, loss_vectors_single_scale, diffusion_model, reconstruction_noise_list, noise_amp = train_single_scale(
            params,
            signals_list,
            fs_list,
            diffusion_models,
            noise_amp_list,
            energy_list,
            reconstruction_noise_list)

        fake_sound = output_signals_single_scale['fake_signal'].squeeze()
        filename = 'fake@%dHz.wav' % params.fs_list[scale_idx]
        write_signal(os.path.join(params.output_folder, filename), fake_sound,
                     params.fs_list[scale_idx], overwrite=False)

        reconstructed_sound = output_signals_single_scale['reconstructed_signal'].squeeze()
        filename = 'reconstructed@%dHz.wav' % params.fs_list[scale_idx]
        write_signal(os.path.join(params.output_folder, filename),
                     reconstructed_sound, params.fs_list[scale_idx], overwrite=False)
        torch.save(reconstruction_noise_list,
                   os.path.join(params.output_folder, 'reconstruction_noise_list.pt'))

        diffusion_models.append(diffusion_model)
        noise_amp_list.append(noise_amp)
        output_signals.append(output_signals_single_scale)
        loss_vectors.append(loss_vectors_single_scale)

    return output_signals, loss_vectors, diffusion_models, noise_amp_list, energy_list, reconstruction_noise_list


def train_single_scale(params, signals_list, fs_list, diffusion_models, noise_amp_list, energy_list,
                       reconstruction_noise_list):
    n_scales = len(params.scales)
    current_scale = n_scales - len(diffusion_models) - 1
    scale_idx = n_scales - current_scale - 1
    input_signal = signals_list[scale_idx].to(params.device)
    params.current_fs = fs_list[scale_idx]
    N = len(input_signal)

    # Create inputs
    real_signal = input_signal.reshape(1, 1, N)

    params.hidden_channels = params.hidden_channels_init if scale_idx == 0 else int(
        params.hidden_channels_init * params.growing_hidden_channels_factor)

    # Initialize diffusion model
    diffusion_model = DiffusionModel(params).to(params.device)

    # Initialize other variables
    pad_size = calc_pad_size(params)
    signal_padder = nn.ConstantPad1d(pad_size, 0)
    reconstruction_noise = get_noise(params, real_signal.shape)

    # Initialize optimizers
    optimizer = optim.Adam(diffusion_model.parameters(), lr=params.learning_rate)

    # Initialize error vectors
    v_rec_loss = np.zeros(params.num_epochs, )

    epochs_start_time = time.time()

    for epoch_num in range(params.num_epochs):
        print_progress = epoch_num % 100 == 0

        # Prepare inputs
        input_noise = get_noise(params, real_signal.shape)
        input_noise = signal_padder(input_noise)

        # Forward pass
        output = diffusion_model((input_noise + real_signal).detach(), real_signal)
        reconstructed_signal = output.detach()

        # Compute reconstruction loss
        rec_loss = torch.mean((real_signal - reconstructed_signal) ** 2)

        # Backpropagation
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

        if print_progress:
            print('[%d/%d] Reconstruction Loss: %.4f' % (epoch_num, params.num_epochs, rec_loss.item()))

        if params.plot_losses:
            v_rec_loss[epoch_num] = rec_loss.item()

    epochs_stop_time = time.time()
    runtime_msg = 'Total time in scale %d: %d[sec] (%.2f[sec]/epoch on avg.). Reconstruction Loss: %.4f' % (
        current_scale, epochs_stop_time - epochs_start_time,
        (epochs_stop_time - epochs_start_time) / params.num_epochs,
        rec_loss.item())
    print(runtime_msg)
    with open(os.path.join(params.output_folder, 'log.txt'), 'a') as f:
        f.write('\n%s\n' % runtime_msg)

    # Save diffusion model
    torch.save(diffusion_model.state_dict(), '%s/diffusion_model_scale_%d.pth' % (params.output_folder, scale_idx))

    # Pack outputs
    if params.plot_losses:
        loss_vectors = {'v_rec_loss': v_rec_loss}
    else:
        loss_vectors = []
    reconstructed_signal = reconstructed_signal.detach().cpu().numpy()[:, 0, :]
    output_signals = {'reconstructed_signal': reconstructed_signal}
    del real_signal, input_noise, reconstructed_signal, output
    diffusion_model.eval()
    if params.is_cuda:
        torch.cuda.empty_cache()
    print('*' * 30 + ' Finished working on scale ' + str(current_scale) + ' ' + '*' * 30)
    return output_signals, loss_vectors, diffusion_model, reconstruction_noise_list, params.initial_noise_amp


def calc_pad_size(params):
    # Calculate padding size based on dilation factors and filter size
    return int((params.filter_size - 1) * sum(params.dilation_factors) / 2)


def get_noise(params, shape):
    # Generate Gaussian noise
    return torch.randn(shape, device=params.device)


# Add other helper functions and parameters here
