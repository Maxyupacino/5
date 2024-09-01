import numpy as np
import matplotlib.pyplot as plt

# a) 从文件中读取数据，并绘制太阳黑子数量随时间变化的图表
data = np.loadtxt("sunspots.txt")
months = data[:, 0]
sunspots = data[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(months, sunspots, color='blue')
plt.title('sunspots')
plt.xlabel('months')
plt.ylabel('the number of sunspots')
plt.grid(True)
plt.show()

# 计算平均周期
print('估算的平均周期为：')
print(500/4)

# b) 计算太阳黑子数据的傅里叶变换，并绘制功率谱图
from cmath import exp, pi

def dft(y):
    N = len(y)
    c = np.zeros(N//2 + 1, complex)
    for k in range(N//2 + 1):
        for n in range(N):
            c[k] += y[n] * exp(-2j * pi * k * n / N)
    return c

dft_result = dft(sunspots)
power_spectrum = abs(dft_result)**2 / len(sunspots)**2

freq = np.fft.fftfreq(len(sunspots))

plt.figure(figsize=(10, 6))
plt.plot(freq[:len(freq)//2], power_spectrum[:len(freq)//2], color='red')
plt.title('Power Spectrum')
plt.xlabel('Frequency (cycles per month)')
plt.ylabel('|Q|^2')
plt.grid(True)
plt.show()

# c)
# 找到功率谱图中第二高峰值对应的近似 k 值，并确定相应正弦波的周期
sorted_indices = np.argsort(power_spectrum)[::-1]
second_peak_index = sorted_indices[1]  # 0 对应最高峰值，1 对应第二高峰值
second_peak_freq = freq[second_peak_index]

# 检查是否出现除零的情况
if second_peak_freq != 0:
    # 计算对应峰值频率的周期
    period = 1 / np.abs(second_peak_freq)
    print("功率谱图中第二高峰值对应的近似频率:", second_peak_freq)
    print("具有此频率的正弦波的周期（以月为单位）:", period)
else:
    print("无法计算周期，因为第二高峰值频率为零。")
