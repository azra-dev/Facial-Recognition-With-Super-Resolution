# Operating System ⚙️
The Rock Pi 5 uses any Linux-based OS with any Desktop Environment. Recommended to use __Debian 12__ as distro and __KDE Plasma__ as DE.

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
## **In-Terminal**

sudo apt-get update

sudo apt-get install build-essential

sudo apt-get install libgtk-3-dev

sudo apt-get install libboost-all-dev

sudo apt install libmpv-dev mpv

## **Virtual  Environment**

# For Super-Resolution

_Note: Make sure to clone https://github.com/TencentARC/GFPGAN first in a separate repository. Paste **requirements.txt**, **VERSION**, **setup.py**, **experiments** folder, and **gfpgan** folder into this repository afterwards. After installation, it can be deleted._

pip install -r requirements.txt
pip install basicsr>=1.4.2 facexlib>=0.2.5
python setup.py develop
pip install realesrgan
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models

# For Face Detection

pip install pybind11 cmake==3.25.2

pip install dlib==19.24.2 face-recognition 

# For GUI

pip install flet==0.19.0
