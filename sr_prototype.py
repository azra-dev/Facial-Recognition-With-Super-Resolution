import cv2
import numpy as np
import os
import torch
import glob
from basicsr.utils import imwrite
from gfpgan import GFPGANer

class SR():
    # ATTRIBUTES
    restorer = None
    mode = 'single'

    def __init__(self,
                input = 'captures/standard', 
                output = 'captures/enhanced',
                version = '1.3',
                upscale = 2,
                bg_upsampler = 'realesrgan',
                bg_tile = 0,
                prefix = None,
                suffix = None,
                only_center_face = None,
                aligned = None,
                ext = 'auto',
                weight = 0.5
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
        self.weight = weight
    
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
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'

        if self.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif self.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif self.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            if os.path.exists("experiments/GFPGANv1.3.pth"):
                url = "experiments/GFPGANv1.3.pth"
            else:
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif self.version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {self.version}.')
        
        model_path = os.path.join('experiments', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        # SET RESTORER
        restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=self.bg_upsampler)
        
        self.restorer = restorer
    
    # UPSCALE / RESTORE
    def Upscale(self):
        if self.input.endswith('/'):
            self.input = self.input[:-1]
        if os.path.isfile(self.input):
            img_list = [self.input]
        else:
            img_list = sorted(glob.glob(os.path.join(self.input, '*')))

        os.makedirs(self.output, exist_ok=True)

        for img_path in img_list:
            # read image
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                input_img,
                has_aligned=self.aligned,
                only_center_face=self.only_center_face,
                paste_back=True,
                weight=self.weight)

            # save restored img
            if restored_img is not None:
                if self.ext == 'auto':
                    extension = ext[1:]
                else:
                    extension = self.ext

                if (self.suffix is None and self.prefix is None):
                    save_restore_path = os.path.join(self.output, f'{basename}.{extension}')
                    print(f"if case name: {save_restore_path}")
                else:
                    restore_name = f'{basename}'
                    if self.suffix is not None and self.prefix is None:
                        restore_name = os.path.join(self.output, f'{restore_name}{self.suffix}.{extension}')
                        print(f"1st case name: {restore_name}")
                    elif self.prefix is not None and self.suffix is None:
                        restore_name = os.path.join(self.output, f'{self.prefix}_{restore_name}.{extension}')
                        print(f"2nd case name: {restore_name}")
                    else:
                        restore_name = os.path.join(self.output, f'{self.prefix}_{restore_name}_{self.suffix}.{extension}')
                        print(f"3rd case name: {restore_name}")
                    save_restore_path = restore_name
                imwrite(restored_img, save_restore_path)
    
    def Run(self):
        self.run_background_sampler()
        self.configure_restorer()
        self.Upscale()
            


def main():
    sr_prototype = SR(
        input = 'captures/standard/Test', 
        output = 'captures/enhanced/Test',
        suffix='H',
        upscale=2)
    sr_prototype.Run()
    del sr_prototype

if __name__ == '__main__':
    main()