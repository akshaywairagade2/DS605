1. Create a conda virtual environment and activate it:
conda create -n ssf python=3.7 -y
conda activate ssf

2. Install CUDA==10.1 with cudnn7 following the https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

3. Install PyTorch==1.7.1 and torchvision==0.8.2 with CUDA==10.1:
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

4. Install timm==0.6.5:
pip install timm==0.6.5

5. Install other requirements:
pip install -r requirements.txt

Data preparation

1. CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

2. For ImageNet-1K, download it from http://image-net.org/, and move validation images to labeled sub-folders. The file structure should look like:

$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...

Fine-tuning a pre-trained model via SSF

1. To fine-tune a pre-trained ViT model via SSF on CIFAR-100 or ImageNet-1K, run:
bash train_scripts/vit/cifar_100/train_ssf.sh

Robustness & OOD

1. To evaluate the performance of fine-tuned model via SSF on Robustness & OOD, run:
bash train_scripts/vit/imagenet_a(r, c)/eval_ssf.sh

