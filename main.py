import os

import cv2
import numpy as np
import polanalyser as pa
from scipy import fftpack


def fourier_lowpass_filter_5x5(image):
    """
    使用傅立叶变换实现5×5窗口的低通滤波

    Args:
        image: 输入图像（假设为5×5 patches组成的图像）

    Returns:
        滤波后的图像
    """
    # 傅立叶变换
    f = fftpack.fft2(image)
    f_shift = fftpack.fftshift(f)

    # 获取图像尺寸
    rows, cols = image.shape

    # 创建5×5低通滤波器掩膜（理想低通滤波器）
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # 设置5×5窗口内的频率成分通过
    window_size = 2  # 5×5窗口半径
    mask[crow - window_size:crow + window_size + 1,
    ccol - window_size:ccol + window_size + 1] = 1

    # 应用滤波器
    f_filtered = f_shift * mask

    # 逆傅立叶变换
    f_ishift = fftpack.ifftshift(f_filtered)
    img_filtered = np.real(fftpack.ifft2(f_ishift))

    return img_filtered


def gaussian_lowpass_filter_5x5(image):
    """
    使用高斯低通滤波器的傅立叶变换实现

    Args:
        image: 输入图像

    Returns:
        滤波后的图像
    """
    # 傅立叶变换
    f = fftpack.fft2(image)
    f_shift = fftpack.fftshift(f)

    # 创建高斯低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 生成坐标网格
    y, x = np.ogrid[:rows, :cols]
    # 调整坐标原点到中心
    y = y - crow
    x = x - ccol

    # 高斯低通滤波器参数
    D0 = 10  # 截止频率
    H = np.exp(-(x ** 2 + y ** 2) / (2 * D0 ** 2))

    # 应用滤波器（限制在5×5区域内）
    mask_5x5 = np.zeros_like(H)
    window_size = 2
    mask_5x5[crow - window_size:crow + window_size + 1,
    ccol - window_size:ccol + window_size + 1] = 1

    H = H * mask_5x5
    f_filtered = f_shift * H

    # 逆傅立叶变换
    f_ishift = fftpack.ifftshift(f_filtered)
    img_filtered = np.real(fftpack.ifft2(f_ishift))

    return img_filtered


if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    image_path = os.path.join(directory, "image/dragon.png")

    img_row = cv2.imread(image_path,-1)
    if img_row is None:
        print("Image not found")

    # 获取四个角度的偏振图像
    img_demosaiced_list = pa.demosaicing(img_row,pa.COLOR_PolarMono)
    img_000, img_045, img_090, img_135 = img_demosaiced_list

    # 计算stokes参数
    img_stokes = pa.calcStokes(img_demosaiced_list,np.deg2rad([0,45,90,135]))
    img_s0 = img_stokes[..., 0]
    img_s1 = img_stokes[..., 1]
    img_s2 = img_stokes[..., 2]
    img_intensity = pa.cvtStokesToIntensity(img_stokes)  # same as s0
    img_dolp = pa.cvtStokesToDoLP(img_stokes)  # [0, 1]
    img_aolp = pa.cvtStokesToAoLP(img_stokes)  # [0, pi]

    theta_airlight = np.arctan(np.sum(img_s2) / np.sum(img_s1)) / 2

    #对偏振图像进行滤波
    img_000_filtered = fourier_lowpass_filter_5x5(img_000)
    img_045_filtered = fourier_lowpass_filter_5x5(img_045)
    img_090_filtered = fourier_lowpass_filter_5x5(img_090)
    img_135_filtered = fourier_lowpass_filter_5x5(img_135)
    img_filtered_stokes = pa.calcStokes([img_000_filtered, img_045_filtered, img_090_filtered, img_135_filtered], np.deg2rad([0, 45, 90, 135]))
    img_filtered_dolp = pa.cvtStokesToDoLP(img_filtered_stokes)  # [0, 1]
    img_filtered_aolp = pa.cvtStokesToAoLP(img_filtered_stokes)  # [0, pi]

    img_filtered_dolp_max = np.max(img_filtered_dolp)

    airlight = (img_090 - img_s0 *(1 - img_dolp) / 2 ) /img_filtered_dolp_max * np.sin(theta_airlight) ** 2
    airlight_infty = 1.02 * airlight

    img_recover = (img_row - airlight) / (1 - airlight / airlight_infty)






