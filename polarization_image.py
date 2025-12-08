import numpy as np
import polanalyser as pa
from typing import Callable


class PolarizationImage:
    """
    处理偏振图像的类，用于去雾和其他偏振图像处理任务
    """
    def __init__(self, image):
        """
        初始化偏振图像处理对象
        
        Args:
            image: 原始偏振图像数据
        """
        if image is None:
            raise ValueError("输入图像不能为空")
        
        self.img_row = image
        self.img_demosaiced_list = pa.demosaicing(self.img_row, pa.COLOR_PolarMono)
        self.img_stokes = pa.calcStokes(self.img_demosaiced_list, np.deg2rad([0, 45, 90, 135]))
        self.aop = pa.cvtStokesToAoLP(self.img_stokes)  # Angle of Polarization
        self.dop = pa.cvtStokesToDoLP(self.img_stokes)  # Degree of Polarization

    def handle_demosaiced_list(self, func: Callable[[np.ndarray], np.ndarray]):
        """
        对去马赛克后的图像列表应用函数并更新相关属性
        
        Args:
            func: 应用于去马赛克图像列表的函数
        """
        if not callable(func):
            raise TypeError("func 必须是可调用的对象")
            
        # self.img_demosaiced_list 每个项目执行func并更新自己
        self.img_demosaiced_list = [func(item) for item in self.img_demosaiced_list]

        self.img_stokes = pa.calcStokes(self.img_demosaiced_list, np.deg2rad([0, 45, 90, 135]))
        self.aop = pa.cvtStokesToAoLP(self.img_stokes)
        self.dop = pa.cvtStokesToDoLP(self.img_stokes)
        return self
        
    def get_stokes_images(self):
        """
        获取Stokes参数图像
        
        Returns:
            tuple: 包含S0, S1, S2参数的图像元组
        """
        img_s0 = self.img_stokes[..., 0]
        img_s1 = self.img_stokes[..., 1]
        img_s2 = self.img_stokes[..., 2]
        return img_s0, img_s1, img_s2

    def get_stokes_images(self):
        """
        获取Stokes参数图像

        Returns:
            tuple: 包含S0, S1, S2参数的图像元组
        """
        img_s0 = self.img_stokes[..., 0]
        img_s1 = self.img_stokes[..., 1]
        img_s2 = self.img_stokes[..., 2]
        return img_s0, img_s1, img_s2

    def get_stokes_image(self,x: int):
        """
        获取Stokes参数图像


        Returns:
            tuple: 包含S0, S1, S2参数的图像元组
        """
        if x >= len(self.img_stokes):
            raise IndexError("索引超出范围")
        return self.img_stokes[..., x]

    def get_img_demosaiced_list(self):
        """
        获取去马赛克后的图像列表

        Returns:
            list: 去马赛克后的图像列表
        """
        return self.img_demosaiced_list

    def get_img_demosaiced(self, x: int):
        """
        获取去马赛克后的图像

        Returns:
            numpy.ndarray: 去马赛克后的图像
        """
        if x >= len(self.img_demosaiced_list):
            raise IndexError("索引超出范围")
        return self.img_demosaiced_list[x]
        
    def get_intensity_and_polarization(self):
        """
        获取强度和偏振信息
        
        Returns:
            tuple: 包含强度、偏振度和偏振角的图像元组
        """
        intensity = pa.cvtStokesToIntensity(self.img_stokes)
        dolp = pa.cvtStokesToDoLP(self.img_stokes)
        aolp = pa.cvtStokesToAoLP(self.img_stokes)
        return intensity, dolp, aolp