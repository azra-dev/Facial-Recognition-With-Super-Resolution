import cv2
import numpy as np
import os
import torch
from gfpgan import GFPGANer

class SR():
    # ATTRIBUTES
    arch = 'original'
    channel_multiplier = 1
    model_name = 'GFPGANv1'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    restorer = None

    def __init__(self,
                input = 'tpdne_dataset/LR128',
                output = 'results',
                version = 1.3,
                upscale = 2,
                bg_upsampler = 'realesrgan',
                bg_tile = 400,
                prefix = 'HR_',
                suffix = None,
                only_center_face = None,
                aligned = None,
                ext = 'auto'
                ):
        self.input = input
        self.output = output
        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.prefix = prefix
        self.suffix = suffix
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.ext = ext
    
    # BACKGROUND SAMPLER
    def run_background_sampler(self):
        if self.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                self.bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=self.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=False # need to set False in CPU mode
                )  
        else:
            self.bg_upsampler = None
    
    # CONFIGURE RESTORER
    def configure_restorer(self):
        # SET VERSION
        if self.version == '1':
            self.arch = 'original'
            self.channel_multiplier = 1
            self.model_name = 'GFPGANv1'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif self.version == '1.2':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANCleanv1-NoCE-C2'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif self.version == '1.3':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.3'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif self.version == '1.4':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.4'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.version == 'RestoreFormer':
            self.arch = 'RestoreFormer'
            self.channel_multiplier = 2
            self.model_name = 'RestoreFormer'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {self.version}.')
        
        model_path = os.path.join('experiments/pretrained_models', self.model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', self.model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = self.url

        # SET RESTORER
        restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler)
        
        self.restorer = restorer
    
    # UPSCALE / RESTORE
    def upscale(self):
        print('lorem ipsum dolor sit amet muna pre')
        


def main():
    sr_prototype = SR()

if __name__ == '__main__':
    main()