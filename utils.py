import os
import numpy as np
import soundfile as sf
import glob
from numpy.fft import fft, ifft
from params import Params
import torch
import torch.nn as nn
import librosa
from scipy import interpolate
from models.diffusionmodel import DiffusionModel
import soundfile as sf

def get_noise(params, shape):
    return torch.randn(shape, device=params.device)

def calc_snr(est, real):
    min_len = min(len(est), len(real))
    real = real[:min_len]
    est = est[:min_len]
    real_fit = real
    est_fit = est
    snr = 10 * np.log10(sum(real_fit ** 2) / sum((est_fit - real_fit) ** 2))
    return snr

def calc_lsd(est, real, eps=1e-15):
    WIN_SIZE = 2048
    min_length = min(len(est), len(real))
    assert abs(len(real) - len(est)) / min_length < 0.2, 'Mismatch in length between 2 signals'
    real = real[:min_length]
    est = est[:min_length]
    X = abs(librosa.stft(est, n_fft=WIN_SIZE, hop_length=WIN_SIZE)) ** 2
    X[X < eps] = eps
    X = np.log(X)
    Y = abs(librosa.stft(real, n_fft=WIN_SIZE, hop_length=WIN_SIZE)) ** 2
    Y[Y < eps] = eps
    Y = np.log(Y)
    Z = (X - Y) ** 2
    lsd = np.sqrt(Z.mean(0)).mean()
    return lsd

def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def calc_gradient_penalty(params, netD, real_data, fake_data, LAMBDA, alpha=None, _grad_outputs=None, mask_ratio=None):
    # Gradient penalty method for WGAN
    if alpha is None:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        if torch.cuda.is_available():
            alpha = alpha.cuda(real_data.get_device())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    if _grad_outputs is None:
        _grad_outputs = torch.ones(disc_interpolates.size())
        if torch.cuda.is_available():
            _grad_outputs = _grad_outputs.cuda(real_data.get_device())
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=_grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((mask_ratio * gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    del gradients, interpolates, _grad_outputs, disc_interpolates
    return gradient_penalty

def create_input_signals(params, input_signal, Fs):
    # Performs downscaling for desired scales and outputs list of signals
    signals_list = []
    fs_list = []
    n_scales = len(params.scales)
    set_first_scale = False
    rf = calc_receptive_field(params.filter_size, params.dilation_factors)
    for k in range(n_scales):
        downsample = params.scales[k]
        fs = int(Fs / downsample)
        if downsample == 1:
            coarse_sig = input_signal
        else:
            coarse_sig = torch.Tensor(librosa.resample(input_signal.squeeze().numpy(), orig_sr=Fs, target_sr=fs))
        if params.run_mode == 'inpainting':
            holes_sum = 0
            for hole_idx in params.inpainting_indices:
                holes_sum += hole_idx[1] - hole_idx[0] + 2*rf
            if (holes_sum) / params.Fs * fs > len(coarse_sig):
                    continue
        if params.speech and fs < 500:
            continue
        if params.set_first_scale_by_energy and not params.speech:
            e = (coarse_sig ** 2).mean()
            if e < params.min_energy_th and not set_first_scale:
                continue
        set_first_scale = True
        signals_list.append(coarse_sig)
        assert np.mod(fs, 1) == 0, 'Sampling rate is not integer'
        fs_list.append(int(fs))

        # Write downsampled real sound
        filename = 'real@%dHz.wav' % fs
        write_signal(os.path.join(params.output_folder, filename), coarse_sig.cpu(), fs)

    return signals_list, fs_list

def calc_pad_size(params, dilation_factors=None, filter_size=None):
    if dilation_factors is None:
        dilation_factors = params.dilation_factors
    if filter_size is None:
        filter_size = params.filter_size
    return int(np.ceil(sum(dilation_factors) * (filter_size - 1) / 2))

def calc_receptive_field(filter_size, dilation_factors, Fs=None):
    if Fs is None:
        # in samples
        return (filter_size * dilation_factors[0] + sum(dilation_factors[1:]) * (filter_size - 1))
    else:
        # in [ms]
        return (filter_size * dilation_factors[0] + sum(dilation_factors[1:]) * (filter_size - 1)) / Fs * 1e3

def resample_sig(params, input_signal, orig_fs=None, target_fs=None):
    # Resamples the input signal to the target sampling frequency
    if orig_fs is None or target_fs is None:
        raise ValueError("Both original and target sampling frequencies must be provided.")
    # Insert resampling logic here
    return resampled_signal

def get_input_signal(params):
    # Retrieve input signal from the file specified in params.input_file
    input_file = params.input_file
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    # Load the input signal from the file
    input_signal, Fs = sf.read(input_file)
    return input_signal

def draw_signal(params, models_list, signals_lengths_list, fs_list, noise_amp_list, condition=None, output_all_scales=False):
    generated_signals = []
    
    for i, model in enumerate(models_list):
        # Assuming model.generate_signal() is a method to generate a signal using the diffusion model
        generated_signal = model.generate_signal()
        
        # Optionally, apply condition or noise amplitude to the generated signal
        if condition is not None:
            # Apply condition to the generated signal
            pass
        
        if noise_amp_list is not None:
            # Apply noise amplitude to the generated signal
            pass
        
        generated_signals.append(generated_signal)
    
    if output_all_scales:
        return np.concatenate(generated_signals)
    else:
        return generated_signals[-1]  # Return the signal from the last scale


def cast_general(x):
    # Check the type of x
    if isinstance(x, int):
        casted_value = int(x)
    elif isinstance(x, float):
        casted_value = float(x)
    elif isinstance(x, str):
        try:
            casted_value = int(x)
        except ValueError:
            try:
                casted_value = float(x)
            except ValueError:
                casted_value = x
    else:
        # If x is not int, float, or string, return it as is
        casted_value = x
    
    return casted_value


def params_from_log(path, gpu_num=0):
    # Initialize a Params object
    params = Params()

    # Read the contents of the log file
    with open(path, 'r') as file:
        lines = file.readlines()

    # Extract parameters from the log file
    for line in lines:
        # Split each line into key-value pairs
        key, value = line.strip().split('=')
        # Remove any leading or trailing spaces from the key and value
        key = key.strip()
        value = value.strip()

        # Update the Params object with the extracted parameters
        setattr(params, key, eval(value))

    # Override the gpu_num parameter if provided
    params.gpu_num = gpu_num

    return params


def noise_amp_list_from_log(path):
    # Initialize an empty list to store noise amplitude values
    noise_amp_list = []

    # Read the contents of the log file
    with open(path, 'r') as file:
        lines = file.readlines()

    # Extract noise amplitude values from the log file
    for line in lines:
        # Check if the line contains the noise amplitude value
        if line.startswith('Noise_amp'):
            # Split the line to extract the value
            _, value = line.strip().split('=')
            # Remove any leading or trailing spaces from the value and convert it to float
            noise_amp_list.append(float(value.strip()))

    return noise_amp_list

def override_params(params, params_override):
    # Iterate through the key-value pairs in params_override
    for key, value in vars(params_override).items():
        # Check if the key exists in the params object
        if hasattr(params, key):
            # Update the parameter value in the params object
            setattr(params, key, value)
        else:
            # If the key does not exist, print a warning message
            print(f"Warning: Parameter '{key}' does not exist in the params object.")

    return params

def write_signal(path, signal, fs, overwrite=False, subtype='PCM_16'):
    # Check if the file already exists and overwrite is False
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File '{path}' already exists.")

    # Write the signal to a WAV file
    sf.write(path, signal, fs, subtype=subtype)

    print(f"Signal written to '{path}'.")

# Example usage:
# write_signal("output.wav", signal, fs)

def time_freq_stitch_by_fft(low_signal, high_signal, low_Fs, high_Fs, filt_file=None):
    # Resample the signals if they have different sampling frequencies
    if low_Fs != high_Fs:
        high_signal_resampled = signal.resample(high_signal, len(low_signal))
    else:
        high_signal_resampled = high_signal
    
    # Perform FFT on both signals
    low_fft = np.fft.fft(low_signal)
    high_fft = np.fft.fft(high_signal_resampled)
    
    # Combine the magnitude of the FFT results
    combined_fft = np.maximum(np.abs(low_fft), np.abs(high_fft))
    
    # Inverse FFT to obtain the stitched signal
    stitched_signal = np.fft.ifft(combined_fft).real
    
    return stitched_signal

# Example usage:
# stitched_signal = time_freq_stitch_by_fft(low_signal, high_signal, low_Fs, high_Fs)