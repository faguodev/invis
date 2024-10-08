# InVis 2.0

## Description
InVis is a tool designed for interactive data visualization. It features advanced interactive data embedding algorithms that allow users to dynamically adjust data embeddings by selecting and moving points within the embedding space. This flexibility facilitates a more intuitive exploration of data structures and patterns.

## Installation Instructions

### Prerequisites
To handle the datasets included in this project, Git Large File Storage (Git LFS) is required. Install Git LFS by following the instructions on the [Git LFS website](https://git-lfs.github.com/).  
If you don't wish to install Git LFS, you can also download the datasets manually, or use your own datasets. 

### Clone the Repository
Clone the InVis GitHub repository:

```bash
git clone https://github.com/faguodev/invis.git
```

### Setting Up the Environment
InVis relies on a Conda environment for managing its dependencies. To set up the Conda environment:
1. Ensure that [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed on your system.
2. Navigate to the project directory where the `environment.yml` file is located.
3. Run the following command to create the Conda environment:

```bash
conda env create -f environment.yml
```

#### Windows

If you're using windows, we recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to set up the environment as above. 

To install InVis on Windows natively, please use the `windows_environment.yml` file instead:

```bash
conda env create -f windows_environment.yml
```

After setting up the environment, additionally install [Tensorflow](https://www.tensorflow.org/install):

```bash
conda activate invis2
pip install tensorflow
```

### GPU Support

For most users, it is recommended to follow the installation instructions above to set up InVis to run on the CPU.  
For optional GPU support, you can follow the detailed instructions in the [GPU Installation Guide](./gpu_installation.md).

## Usage
To use InVis, activate the Conda environment and start the application with the following commands:

```bash
conda activate invis2 
python Main.py
```

A detailed user guide is included in the repository which explains how to use the application effectively.

## Credits
InVis was initially developed by Daniel Paurat with algorithmic contributions from Dino Oglic. It was extended by Florian Chen to include iterative optimization methods for efficient embedding adaptations.

## Publications
This software has been used in the following research papers:

- Chen, Florian, and Gärtner, Thomas, "Scalable Interactive Data Visualization," in Proc. ECML-PKDD, Springer, 2024, pp. 429–433.

- Oglic, Dino, Paurat, Daniel, and Gärtner, Thomas, "Interactive Knowledge-Based Kernel PCA," in Proc. ECML-PKDD, Springer, 2014, pp. 501–516.

- Paurat, Daniel, and Gärtner, Thomas, "InVis: A Tool for Interactive Visual Data Analysis," in Proc. ECML-PKDD, Springer, 2013, pp. 672–676.

## Contact Information
For further information, inquiries, or feedback regarding InVis, please feel free to contact us:  
florian.chen [at] tuwien.ac.at
