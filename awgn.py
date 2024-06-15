import numpy as np
import matplotlib.pyplot as plt
def add_awgn_noise(signal, snr_dB):
    """
    在信号中添加高斯白噪声
    signal: 原始信号
    snr_dB: 信噪比（以dB为单位）
    :return: 添加噪声后的信号
    """
    # 计算信号功率
    signal_power = np.mean(np.abs(signal)**2)
    
    # 将信噪比从dB转换为线性值
    snr_linear = 10**(snr_dB / 10)
    
    # 计算噪声功率
    noise_power = signal_power / snr_linear
    
    # 生成高斯白噪声
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    # 添加噪声到原始信号
    noisy_signal=np.array(signal + noise)
    noisy_signal=np.where(noisy_signal<=0.5,0,1)
    return noisy_signal