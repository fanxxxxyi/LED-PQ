# LED-PQ: Learnable Elasticity Dynamics for Pruning-Quantization Joint Learning ([Link](https://xxx))![]( https://xxx).



## Tips

Any problem, please contact the authors via emails: xiaoyifan_xdu@163.com. Do not post issues with github as much as possible, just in case that I could not receive the emails from github thus ignore the posted issues.


## Requirements

- PyTorch 1.0.1
- Python 3.5+

## Running Code

In this code, you can run our models on CIFAR-10 and CIFAR-100 dataset. 


### Rank Generation

```shell
python rank_generation.py \
--resume [pre-trained model dir] \
--arch [model arch name] \
--limit [batch numbers] \
--gpu [gpu_id]

```

### LED-Search for a and b(Use ResNet56 on CIFAR-10 as an example)

```shell
python resnet_main.py \
--resume [pre-trained model dir] \
--arch resnet_56 \
--rankPath ./rank_conv/CIFAR10/resnet_56/
--gpu [gpu_id]
--min_lub True
```

### Model Training

##### 1. VGG-16

```shell
python vgg_main.py \
--resume [pre-trained model dir] \
--arch vgg_16_bn \
--rankPath ./rank_conv/CIFAR10/vgg_16_bn/
--gpu [gpu_id]
--min_lub False
--lub ./log_{}/vgg_16_bn_ea_min.data
```
##### 2. ResNet56

```shell
python resnet_main.py \
--resume [pre-trained model dir] \
--arch resnet_56 \
--rankPath ./rank_conv/CIFAR10/resnet_56/
--gpu [gpu_id]
--min_lub False
--lub ./log_{}/resnet_56_ea_min.data
```
##### 3. MobilenetV2_CIFAR100

```shell
python mbv2_ch_main.py \
--resume [pre-trained model dir] \
--arch MobilenetV2 \
--rankPath ./rank_conv/CIFAR100/MobilenetV2/
--gpu [gpu_id]
--min_lub False
--lub ./log_{}/mobilenetv2_CIFAR100_ea_min.data
```
##### 4. ResNet56_CIFAR100

```shell
python resnet_ch_main.py \
--resume [pre-trained model dir] \
--arch resnet_56 \
--rankPath ./rank_conv/CIFAR100/resnet_56/
--gpu [gpu_id]
--min_lub False
--lub ./log_{}/resnet_56_CIFAR100_ea_min.data
```

### Get BOPs
```shell
python count_bops.py
```

### Other optional arguments
```
optional arguments:
    --data_dir			dataset directory
    				default='./data'
    --dataset			dataset name
    				default: cifar10
    				Optional: cifar10, cifar100
    --lr			initial learning rate
    				default: 0.01
    --lr_decay_step		learning rate decay step
				default: 5,10
    --resume			load the model from the specified checkpoint
    --resume_mask		mask loading directory
    --gpu			Select gpu to use
    				default: 0
    --job_dir			The directory where the summaries will be stored.
    --epochs			The num of epochs to train.
    				default: 400
    --train_batch_size		Batch size for training.
    				default: 128
    --arch			The architecture to prune
    				default: vgg_16_bn
				Optional: vgg_16_bn, resnet_56, resnet_110, densenet_40, mobilenetv2
```

## Pretrained Models
```
Please contact the authors to obtain pretrained models via emails: xiaoyifan_xdu@163.com.
```