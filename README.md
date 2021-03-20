This is the unofficial PyTorch implementation of [Neuron Coverage-Guided Domain Generalization](https://arxiv.org/pdf/2103.00229.pdf).

#### Environment

Python==3.7.2

PyTorch==1.5.1

numpy==1.16.3

#### data

Please download the data from https://drive.google.com/open?id=0B6x7gtvErXgfUU1WcGY5SzdwZVk and use the official train/val split.

```
Neuron Coverage-Guided Domain Generalization
├── data
│   ├── Train val splits and h5py files pre-read
```

#### train

```
python main.py --batch_size 64 --n_classes 7 --learning_rate 0.001 --image_size 256 --nesterov True --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art_painting cartoon photo --target sketch --epochs 100
```

