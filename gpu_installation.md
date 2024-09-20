
---

# Quick Guide to Build and Run InVis Docker Container

## Pre-requisites
Before you begin, ensure the following are set up:

1. **NVIDIA GPU Powered Machine**:
   - Ensure you have a machine with an NVIDIA GPU and Ubuntu installed.

2. **NVIDIA Drivers**:
   - Install NVIDIA drivers version >= 450.80.02 on your machine.

3. **NVIDIA Container Toolkit**:
   - Install the NVIDIA Container Toolkit. Follow the installation guide here: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

4. **xhost**
   - Ensure `xhost` is installed on your system to manage access control to the X server. On Ubuntu, xhost is part of the x11-xserver-utils package. You can install it with:
   ```bash
   sudo apt-get install x11-xserver-utils
   ```

## Steps to Build and Run the Docker Container

### 1. Clone the Repository
First, clone the InVis GitHub repository:

```bash
git clone https://github.com/faguodev/invis.git
```

### 2. Navigate to the Project Folder
Change directory to the project folder:

```bash
cd invis
```

### 3. Build the Docker Image
Execute the `build.sh` script to build the Docker image:

```bash
bash build.sh
```

### 4. Run the Docker Container
Execute the `run.sh` script to start the Docker container:

```bash
bash run.sh
```

### Customizing Container Name
If you want to specify a custom name for the container, you can pass the name as an argument to the `run.sh` script:

```bash
bash run.sh my_custom_container
```

If no argument is provided, the default container name `container01` will be used.

---
