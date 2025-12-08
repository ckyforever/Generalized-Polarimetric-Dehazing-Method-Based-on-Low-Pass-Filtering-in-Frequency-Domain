import os

import cv2
import numpy as np
import polanalyser as pa
from scipy import fftpack

from polarization_image import PolarizationImage


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


def process_polarization_image(img_row):
    """
    处理偏振图像的主要函数
    
    Args:
        img_row: 原始图像数据
        
    Returns:
        tuple: 恢复的图像和相关信息
    """
    # 创建偏振图像处理对象
    polar_img = PolarizationImage(img_row)

    theta_airlight = np.average(polar_img.aop) / 2

    filtered_polar_img = polar_img.handle_demosaiced_list(fourier_lowpass_filter_5x5)

    img_filtered_dolp_max = np.max(filtered_polar_img.dop)

    airlight = (polar_img.get_img_demosaiced(2) - polar_img.get_stokes_image(0) * (1 - polar_img.dop) / 2) / img_filtered_dolp_max * np.sin(theta_airlight) ** 2
    airlight_infty = 1.02 * np.max(airlight)

    img_recover = (img_row - airlight) / (1 - airlight / airlight_infty)
    
    return img_recover, polar_img, filtered_polar_img


if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    image_path = os.path.join(directory, "image/dragon.png")

    img_row = cv2.imread(image_path, -1)
    if img_row is None:
        print("Image not found")
    else:
        # 处理偏振图像
        img_recover, original_polar, filtered_polar = process_polarization_image(img_row)
        cv2.imshow("Original Image", img_row)
        cv2.imshow("Recovered Image", img_recover)
        cv2.waitKey(0)
        print("图像处理完成")