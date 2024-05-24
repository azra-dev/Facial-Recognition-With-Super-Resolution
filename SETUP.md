# Operating System ⚙️
The Rock Pi 5 uses any Linux-based OS with any Desktop Environment. Recommended to use __Debian 12__ as distro and __KDE Plasma__ as DE.

# Setting Up System 🔧
**Make yourself as a root**
> su
> nano /etc/sudoers
> append <username> ALL=(ALL:ALL) ALL

**Get Linux Headers**
> sudo apt install linux-headers-$(uname -r)

**Install Necessities** (either from website or download flatpak and snap in Discover)
> Visual Studio Code
> Discord (optional)

# Dependencies 💿
**Terminal**
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev

**Virtual  Environment**
pip install pybind11
pip install cmake==3.25.2
pip install dlib==19.24.2
pip install face-recognition
