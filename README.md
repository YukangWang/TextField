# TextField: Learning A Deep Direction Field for Irregular Scene Text Detection

## Introduction

The code and trained models of: 

TextField: Learning A Deep Direction Field for Irregular Scene Text Detection, TIP 2019 [[Paper]](https://arxiv.org/abs/1812.01393)

## Citation

Please cite the related works in your publications if it helps your research:

```

@article{xu2018textfield,
  title={TextField: Learning A Deep Direction Field for Irregular Scene Text Detection},
  author={Xu, Yongchao and Wang, Yukang and Zhou, Wei and Wang, Yongpan and Yang, Zhibo and Bai, Xiang},
  journal={arXiv preprint arXiv:1812.01393},
  year={2018}
}

```

## Prerequisite

* Caffe and SynthText pretrained model [[Link]](https://drive.google.com/file/d/1C4EUllZMTNYt_Q2t4PjZypepYjHechvj/view?usp=sharing)

* Datasets: [[Total-Text]](http://www.cs-chan.com/source/ICDAR2017/totaltext.zip), [[ICDAR2015]](http://rrc.cvc.uab.es/?ch=4&com=downloads)

* OpenCV 3.4.3

* MATLAB


## Usage

#### 1. Install Caffe

```bash

cp Makefile.config.example Makefile.config
# adjust Makefile.config (for example, enable python layer)
make all -j16
# make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
make pycaffe

```
Please refer to [Caffe Installation](http://caffe.berkeleyvision.org/install_apt.html) to ensure other dependencies.

#### 2. Data and model preparation

```bash

# download datasets and pretrained model then
mkdir data && mv [your_dataset_folder] data/
mkdir models && mv [your_pretrained_model] models/

```  

#### 3. Training scripts
  

```bash

# an example on Total-Text dataset
cd examples/TextField/
python train.py --gpu [your_gpu_id] --dataset total --initmodel ../../models/synth_iter_800000.caffemodel

```

#### 4. Evaluation scripts

```bash

# an example on Total-Text dataset
cd evaluation/total/
./eval.sh

```

## Results and Trained Models

#### Total-Text

| Recall | Precision | F-measure | Link |
|:-------------:|:-------------:|:-------------:|:-----:|
| 0.816 | 0.824 | 0.820 | [[Google drive]](https://drive.google.com/file/d/1FAiL2C0WOuN5QFSD6wfLSgP29mmBUdIV/view?usp=sharing) |

>*lambda=0.50 for post-processing

#### ICDAR2015

| Recall | Precision | F-measure | Link |
|:-------------:|:-------------:|:-------------:|:-----:|
| 0.811 | 0.846 | 0.828 | [[Google drive]](https://drive.google.com/file/d/1T6lBbe1BXfppsuijZTE_LSaH1XT_5_jD/view?usp=sharing) |

>*lambda=0.75 for post-processing
