# Tong Simulator
A simulator of ISI's robot manipulator system called "Tong system".

## Environment
### Download isaacgym
Install latest isaacgym (too old versions not work)
- Download: https://developer.nvidia.com/isaac-gym/download


### Python environment
Python version: == 3.8 (3.7 might not work)
```
$ cd /path/to/isaacgym/python
$ pip install -e .
$ cd tong_simulator
$ pip install -r requirements.txt
```
- Please install isaacgym first 
- If you want use newer versions of pytorch, you might have to upgrade your gcc version to build gymtorch (but you can use it)

\
If you use conda env, set LD_LIBRARY_PATH environment variable like below so that isaacgym can find libraries
```
$ echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu:~/anaconda3/envs/[env name]/lib' >> ~/.bashrc
```

* /lib/x86_64-linux-gnu is set because some other softwares become not to work in certain environment (e.g. ssh in ubuntu22.04)
    * In ubuntu18.04 or 20.04, it is ok that:
    ```
     $ echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/anaconda3/envs/[env name]/lib' >> ~/.bashrc
    ```

## Usage
### 1. Launch tongsim in a terminal
```
$ python main.py
```
### 2. Run your program that communicates with tongsim in another terminal
- TCP communication
    - Send joint angle targets to 127.0.0.1:5555 (you can change if needed).
    - Recieve observations (camera image, joint angles, force sensors and world states) from 127.0.0.1:5555.
- **Official interface program is avaiable**: https://github.com/crumbyRobotics/tong_system
    - If you like gym env like interface, please use it.