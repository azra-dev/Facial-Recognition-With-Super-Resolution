# Operating System ⚙️
The Raspberry 4 uses any Linux-based OS with any Desktop Environment. Recommended to use __Debian 12__ as distro and __KDE Plasma__ as DE.

# Setting Up System 🔧
**Make yourself as a root**
> su
> 
> nano /etc/sudoers
> 
> append <username> ALL=(ALL:ALL) ALL

**Get Linux Headers**
> sudo apt install linux-headers-$(uname -r)

**Install Necessities** (either from website or download flatpak and snap in Discover)
> Visual Studio Code
> 
> Discord (optional)

# Dependencies 💿
### **In-Terminal**
- sudo apt-get update
- sudo apt-get install build-essential
- sudo apt-get install libgtk-3-dev
- sudo apt-get install libboost-all-dev
- sudo apt install libmpv-dev mpv

### **Virtual  Environment**

#### For Super-Resolution
- pip install -r _gfppgan-package/requirements.txt
- pip install basicsr>=1.4.2 facexlib>=0.2.5
- python _gfpgan-package/setup.py develop
- pip install realesrgan
- wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models

#### For Face Detection
- pip install keras-facenet tensorflow opencv-python mtcnn
> [!WARNING]
> The following libraries may not be used in the thesis.

- pip install pybind11 cmake==3.25.2
- pip install dlib==19.24.2 face-recognition 

#### For GUI
- pip install flet==0.19.0

# Error Handling 🔧

### Git Error
During commit, sometimes an error of `RPC failed; HTTP 400 curl 22 The requested URL returned error: 400 Bad Request` will be shown likely due to committing something with a file size beyond 100MB. Run the command in the terminal to produce a .gitignore file to ignore large file size:
```bash
find . -size +100M | cat >> .gitignore
```
However, do remove the `./` prefix of each generated output to properly address the files to be ignored

### basicsr error
Upon running the sr-prototype.py program, an error will be recieved which comes from basicsr module due to deprication. The module is modifiable thankfully with the following step:

Open `your_venv/lib/python3.11/site-packages/basicsr/data/degradations.py` and on line 8, simply change:
```py
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
to
```py
from torchvision.transforms.functional import rgb_to_grayscale
```