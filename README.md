# Tong Simulator
[Intelligent Systems and Informatics Laboratory](https://www.isi.imi.i.u-tokyo.ac.jp/?lang=ja), The University of Tokyo  
Information: [Ryo Takizawa](https://crumbyrobotics.github.io/)

A simulator for ISI's robot manipulator system, "Tong system".

<video src="demo.mp4" controls loop height="300"></video>

## Environment

### Download Isaac Gym
Install the latest Isaac Gym (older versions may not work).  
Download: [Isaac Gym](https://developer.nvidia.com/isaac-gym/download)

### Python Environment
Python version: **3.8** (3.7 may not work)
```bash
cd /path/to/isaacgym/python
pip install -e .
cd tong_simulator
pip install -r requirements.txt
```
- Install Isaac Gym first.
- For newer PyTorch versions, you may need to upgrade your GCC version to build gymtorch.

If using a conda environment, set the `LD_LIBRARY_PATH` so Isaac Gym can find libraries:
```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu:~/anaconda3/envs/[env name]/lib' >> ~/.bashrc
```
- `/lib/x86_64-linux-gnu` is included to avoid issues with some software (e.g., SSH on Ubuntu 22.04).
- On Ubuntu 18.04 or 20.04, you can use:
    ```bash
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/[env name]/lib' >> ~/.bashrc
    ```

## Usage

### 1. Launch tongsim in a terminal
```bash
python main.py
```

### 2. Run your program that communicates with tongsim in another terminal
- **TCP communication**
    - Send joint angle targets to `127.0.0.1:5555` (configurable).
    - Receive observations (camera image, joint angles, force sensors, world states) from `127.0.0.1:5555`.
- **Official interface program available:** [tong_system](https://github.com/crumbyRobotics/tong_system)
    - For a Gym-like interface, use the official program.
