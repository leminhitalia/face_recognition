# face_recognition

For Windows:
1. Download and install Python for Windows, recommend to download .exe file: https://www.python.org/downloads/
2. Download Cmake for Windows and config Cmake path (EX: C:\Program Files\cmake-3.11.2-win64-x64\bin) in the PATH environment, recommend to download zip file: https://cmake.org/download/
3. Download and install Microsoft Visual C++ Build Tools 2015: https://msdn.microsoft.com/en-us/library/ms235639.aspx


After installing, you can check to see if the soft is good:
- Check python: python -V
- Check Cmake: cmake -version
- Check Microsoft Visual C++ Build Tools 2015: Press Windows button and search 'build tools command prompt', you will see Build Tools Command Prompts of Visual C++ 2015


If everything is good, you can continue to install necessary libraries:
- Install 'opencv' for Python (https://pypi.org/project/opencv-python/): **pip install opencv-python**
- Install 'numpy' for Python (https://pypi.org/project/numpy/): **This is included in above 'opencv' package.**
- Install 'pillow' for Python (https://pypi.org/project/Pillow/): **pip install Pillow**
- Install 'opencv-contrib-python' for Python (https://pypi.org/project/opencv-contrib-python/): **pip install opencv-contrib-python**

IDE:
- Recommend to use IntelliJ Community (https://www.jetbrains.com/idea/download/#section=windows) with installing Python Community Edition plugin (https://confluence.jetbrains.com/display/PYH/)


Don't need to read this:
- The context that the topic is brought up; why and who it matters; what, when, and how it helps for whom:
- What people can take away after joining the topic:
- Main contents of the topic:

pip install Keras

pip install scikit-learn

pip install imutils

pip install argparse

pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl

Install CUDA:
CUDA 9: https://developer.nvidia.com/cuda-downloads
CUDA 8 (need): https://developer.nvidia.com/cuda-80-ga2-download-archive

Install cuDNN: https://developer.nvidia.com/rdp/cudnn-download