# Description

This repository contains the implementation of the FROMAGe model as proposed in the original [paper](https://arxiv.org/pdf/2301.13823.pdf). The code was obtained after cloning the author's github [repo](https://github.com/kohjingyu/fromage) and the purpose of this repository is to explore the possibilities for in-context learning of FROMAGe, by working with different prompting templates and strategies. 

# Instructions for the demos

The `demo_instructions.ipynb` provides some information about the demos. You don't have to download any data manually, because there are already uploaded in this repo. This means that when you clone the repo you already have the data used in inference. Therefore the steps are the following:

1. Clone this repo:
`git clone https://github.com/apostolikas/FROMAGe-Prompting-Strategies.git `
2. Download the cc3m embeddings from this [link](https://drive.google.com/file/d/1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi/view) and place them in ` fromage_model/ ` directory.
3. Install the environment running the `jobs/install_env.job` file.
4. Run the demos


# Instructions for the experiments in each branch

The blogpost.md provides a mini report of the strategies and the experiments conducted in this work. It also contains the results and a discussion of them. 
For your convenience, there are also some demos you can run and get a first experience of how the model works in each task and how some augmentations might actually help it perform even better.

Before you run anything, it is very important to follow the steps below:
1. Clone this repo:
`git clone https://github.com/apostolikas/FROMAGe-Prompting-Strategies.git `
2. Download the Flickr-8k dataset from this [link](https://drive.google.com/drive/folders/1wkQAqNnIPPijeKgyCUEDtWa56OFkugEY?usp=sharing), unzip and put the 2 folders inside the main directory of the project.
3. Download the Guided-vqa dataset from this [link](https://drive.google.com/drive/folders/1wkQAqNnIPPijeKgyCUEDtWa56OFkugEY?usp=sharing), unzip and put the folder inside the main directory of the project.
4. Download the Validation Set from this [link](https://visualdialog.org/data) and put the folder inside the main directory of the project.
5. Make sure to install the environment before running any script.
6. Before running anything make sure to download the cc3m embeddings from this [link](https://drive.google.com/file/d/1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi/view) and place them in ` fromage_model/ ` directory.
7. Run the scripts and explore the FROMAGe's potential on several tasks.

