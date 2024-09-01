import numpy as np
import matplotlib.pyplot as plt

# 从 dow.txt 文件中读取数据
with open("dow.txt", "r") as file:
    data = np.array([float(line.strip()) for line in file])

# 绘制原始数据的图表
plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data")

# 计算数据的离散傅里叶变换的系数
fft_coefficients = np.fft.rfft(data)

# 将系数数组中除了前10%的元素都设置为零
fft_coefficients_truncated_10 = fft_coefficients.copy()
cutoff_index_10 = int(len(fft_coefficients_truncated_10) * 0.1)
fft_coefficients_truncated_10[cutoff_index_10:] = 0

# 计算截断后系数的逆傅里叶变换，并绘制重建数据
inverse_fft_10 = np.fft.irfft(fft_coefficients_truncated_10)
plt.plot(inverse_fft_10, label="Reconstructed Data (10% Coefficients Kept)")

# 将所有傅里叶系数中除了前2%的元素都设置为零
fft_coefficients_truncated_2 = fft_coefficients.copy()
cutoff_index_2 = int(len(fft_coefficients_truncated_2) * 0.02)
fft_coefficients_truncated_2[cutoff_index_2:] = 0

# 计算截断后系数的逆傅里叶变换，并绘制重建数据
inverse_fft_2 = np.fft.irfft(fft_coefficients_truncated_2)
plt.plot(inverse_fft_2, label="Reconstructed Data (2% Coefficients Kept)")

# 图表设置
plt.legend()
plt.xlabel("Day")
plt.ylabel("Dow Jones Industrial Average")
plt.title("Original Data vs Reconstructed Data")
plt.show()