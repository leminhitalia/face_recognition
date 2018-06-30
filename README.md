# face_recognition

## Set up environment for Windows: <br />
1. Install Python 5: https://www.python.org/ftp/python/3.5.0/python-3.5.0-amd64.exe <br />
2. Download Cmake for Windows and config Cmake path (EX: C:\Program Files\cmake-3.11.2-win64-x64\bin) in the PATH environment, recommend to download zip file: https://cmake.org/download/ <br />
3. Download and install Microsoft Visual C++ Build Tools 2015: https://msdn.microsoft.com/en-us/library/ms235639.aspx

## After installing, you can check to see if the environment and libraries are good: <br />
- Check python: python -V <br />
- Check Cmake: cmake -version <br />
- Check Microsoft Visual C++ Build Tools 2015: Press Windows button and search 'build tools command prompt', you will see Build Tools Command Prompts of Visual C++ 2015

## IDE: <br />
- Recommend to use IntelliJ Community (https://www.jetbrains.com/idea/download/#section=windows) with installing Python Community Edition plugin (https://confluence.jetbrains.com/display/PYH/)


## Install libraries: <br />
pip install -r requirements.txt <br />
OR <br />
pip install opencv-python <br />
pip install dlib <br />
pip install Keras <br />
pip install scikit-learn <br />
pip install imutils <br />
pip install argparse <br />
pip3 install --upgrade tensorflow (Admin CMD, upgrade pip version: python -m pip install --upgrade pip) <br />
pip3 install --upgrade tensorflow-gpu (Admin CMD, upgrade pip version: python -m pip install --upgrade pip, require CUDA and cuDNN) <br />

## For Tensorflow:
- Tensorflow 1.8.0 may not work on Python 3.6, therefore, we should use Python 3.5 (see "Set up environment for Windows") <br />
- Install CUDA and cuDNN: Check compatibility versions (Tested source configurations) https://www.tensorflow.org/install/install_sources
    - CUDA 9: https://developer.nvidia.com/cuda-downloads <br />
    - CUDA 8: https://developer.nvidia.com/cuda-80-ga2-download-archive
    - CUDA other version: Please find on Internet
    - cuDNN: https://developer.nvidia.com/rdp/cudnn-download (must register a developer account on this site)
- Other resources to refer to Tensorflow:
    - https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support <br />
    - https://www.tensorflow.org/install/install_sources#tested_source_configurations <br />
    - https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package <br />
    - pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl <br />
    - pip install --upgrade https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-1.8.0-py3-none-any.whl


Don't need to read this:
- The context that the topic is brought up; why and who it matters; what, when, and how it helps for whom:
- What people can take away after joining the topic:
- Main contents of the topic: