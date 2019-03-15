#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import caffe
import argparse
import numpy as np

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, lr=1):
    """
    Def conv layer with ReLU. bottom: input blob; nout: output channel.
    """
    conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, 
        param=[dict(lr_mult=1*lr, decay_mult=1), dict(lr_mult=2*lr, decay_mult=0)])
    return conv, caffe.layers.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2, pad=0):
    """
    Def max pooling layer.
    """
    pool = caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)
    return pool

def write_net(dataset, patch_size, npRatio):
    """
    Generate train.prototxt and test.prototxt.
    Dataset: 'synth' for Synth-Text; 'ctw' for SCUT-CTW1500; 'total' for Total-Text; 'ic15' for ICDAR2015; 'td' for MSRA-TD500.
    """
    net = caffe.NetSpec()
    datalayer_params = dict(data_dir='../../data/', dataset=dataset, patch_size=patch_size, seed=123, mean=(103.939, 116.779, 123.68))
    losslayer_params = dict(npRatio=npRatio)
    net.image, net.vec, net.weight = caffe.layers.Python(module='pylayerUtils', layer='DataLayer', ntop=3, param_str=str(datalayer_params))

    net.conv1_1, net.relu1_1 = conv_relu(net.image, 64)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 64)
    net.pool1 = max_pool(net.relu1_2)

    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 128)
    net.pool2 = max_pool(net.relu2_2)

    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 256)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 256)
    net.pool3 = max_pool(net.relu3_3)

    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 512)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 512)
    net.pool4 = max_pool(net.relu4_3)

    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 512)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 512)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 512)

    net.sconv5_1, net.srelu5_1 = conv_relu(net.relu5_3, 512, ks=5, pad=2, lr=10)
    net.sconv5_2, net.srelu5_2 = conv_relu(net.srelu5_1, 512, ks=1, pad=0, lr=10)
    net.sconv5_3 = caffe.layers.Convolution(net.srelu5_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    net.srelu5_3 = caffe.layers.ReLU(net.sconv5_3, in_place=True)
    net.ups5 = caffe.layers.Deconvolution(net.srelu5_3,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])

    net.sconv4_1, net.srelu4_1 = conv_relu(net.relu4_3, 512, ks=5, pad=2, lr=10)
    net.sconv4_2, net.srelu4_2 = conv_relu(net.srelu4_1, 512, ks=1, pad=0, lr=10)
    net.sconv4_3 = caffe.layers.Convolution(net.srelu4_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    net.srelu4_3 = caffe.layers.ReLU(net.sconv4_3, in_place=True)
    net.ups4 = caffe.layers.Deconvolution(net.srelu4_3,convolution_param=dict(num_output=256, kernel_size=4, stride=2, pad=1, bias_term=False), param=[dict(lr_mult=0)])

    net.sconv3_1, net.srelu3_1 = conv_relu(net.relu3_3, 256, ks=5, pad=2, lr=10)
    net.sconv3_2, net.srelu3_2 = conv_relu(net.srelu3_1, 256, ks=1, pad=0, lr=10)
    net.sconv3_3 = caffe.layers.Convolution(net.srelu3_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.0001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    net.srelu3_3 = caffe.layers.ReLU(net.sconv3_3, in_place=True)

    bottom_layers = [net.srelu3_3, net.ups4, net.ups5]
    net.sconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.fconv1, net.frelu1 = conv_relu(net.sconcat, 512, ks=1, pad=0, lr=10)
    net.fconv2, net.frelu2 = conv_relu(net.frelu1, 512, ks=1, pad=0, lr=10)
    net.fconv3 = caffe.layers.Convolution(net.frelu2, kernel_size=1, stride=1, num_output=2, pad=0,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])

    net.fup = caffe.layers.Deconvolution(net.fconv3,convolution_param=dict(num_output=2, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.loss = caffe.layers.Python(net.fup, net.vec, net.weight, module='pylayerUtils', layer='EuclideanLossLayerWithOHEM', loss_weight=1, param_str=str(losslayer_params))

    with open('train.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
    with open('test.prototxt', 'w') as f:
        f.write(str(net.to_proto()))

def write_deploy():
    """
    Generate deploy.prototxt.
    """
    net = caffe.NetSpec()
    net.data = caffe.layers.Input(input_param={'shape':{'dim':[1,3,512,512]}})

    net.conv1_1, net.relu1_1 = conv_relu(net.data, 64)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 64)
    net.pool1 = max_pool(net.relu1_2)

    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 128)
    net.pool2 = max_pool(net.relu2_2)

    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 256)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 256)
    net.pool3 = max_pool(net.relu3_3)

    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 512)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 512)
    net.pool4 = max_pool(net.relu4_3)

    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 512)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 512)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 512)

    net.sconv5_1, net.srelu5_1 = conv_relu(net.relu5_3, 512, ks=5, pad=2)
    net.sconv5_2, net.srelu5_2 = conv_relu(net.srelu5_1, 512, ks=1, pad=0)
    net.sconv5_3 = caffe.layers.Convolution(net.srelu5_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu5_3 = caffe.layers.ReLU(net.sconv5_3, in_place=True)
    net.ups5 = caffe.layers.Deconvolution(net.srelu5_3,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])

    net.sconv4_1, net.srelu4_1 = conv_relu(net.relu4_3, 512, ks=5, pad=2)
    net.sconv4_2, net.srelu4_2 = conv_relu(net.srelu4_1, 512, ks=1, pad=0)
    net.sconv4_3 = caffe.layers.Convolution(net.srelu4_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu4_3 = caffe.layers.ReLU(net.sconv4_3, in_place=True)
    net.ups4 = caffe.layers.Deconvolution(net.srelu4_3,convolution_param=dict(num_output=256, kernel_size=4, stride=2, pad=1, bias_term=False), param=[dict(lr_mult=0)])

    net.sconv3_1, net.srelu3_1 = conv_relu(net.relu3_3, 256, ks=5, pad=2)
    net.sconv3_2, net.srelu3_2 = conv_relu(net.srelu3_1, 256, ks=1, pad=0)
    net.sconv3_3 = caffe.layers.Convolution(net.srelu3_2, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.0001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu3_3 = caffe.layers.ReLU(net.sconv3_3, in_place=True)

    bottom_layers = [net.srelu3_3, net.ups4, net.ups5]
    net.sconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.fconv1, net.frelu1 = conv_relu(net.sconcat, 512, ks=1, pad=0)
    net.fconv2, net.frelu2 = conv_relu(net.frelu1, 512, ks=1, pad=0)
    net.fconv3 = caffe.layers.Convolution(net.frelu2, kernel_size=1, stride=1, num_output=2, pad=0,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.fup = caffe.layers.Deconvolution(net.fconv3,convolution_param=dict(num_output=2, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])

    with open('deploy.prototxt', 'w') as f:
        f.write(str(net.to_proto()))

def write_solver(base_lr, iters, snapshot):
    """
    Generate solver.prototxt.
    base_lr: learning rate;
    iters: max iterations;
    snapshot: the prefix of saved models.
    """
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()
    sovler_string.train_net = 'train.prototxt'
    sovler_string.test_net.append('test.prototxt')
    sovler_string.test_iter.append(500)
    sovler_string.test_interval = 999999
    sovler_string.type = 'Adam'
    sovler_string.base_lr = base_lr
    sovler_string.momentum = 0.9
    sovler_string.momentum2 = 0.999
    sovler_string.lr_policy = 'fixed'
    sovler_string.display = 100
    sovler_string.average_loss = 100
    sovler_string.max_iter = iters
    sovler_string.snapshot = 10000
    sovler_string.snapshot_prefix = snapshot
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
    sovler_string.test_initialization = 0

    with open('solver.prototxt', 'w') as f:
        f.write(str(sovler_string))

def train(initmodel, gpu):
    """
    Train the net.
    """
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    solver = caffe.AdamSolver('solver.prototxt')
    if initmodel:
        solver.net.copy_from(initmodel)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp(solver.net, interp_layers)
    solver.step(solver.param.max_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset name.")
    parser.add_argument("--initmodel", help="Init caffemodel.")
    parser.add_argument("--gpu", required=True, type=int, help="Device ids.")
    args = parser.parse_args()

    if not os.path.isdir('snapshot_1e-4'):
        os.makedirs('snapshot_1e-4')
    if not os.path.isdir('snapshot_1e-5'):
        os.makedirs('snapshot_1e-5')

#    write_net('synth', 384, 3)
#    write_solver(1e-5, 800000, 'snapshot_1e-4/synth')
#    train(args.initmodel, args.gpu)

    write_net(args.dataset, 384, 3)
    write_solver(1e-5, 160000, 'snapshot_1e-4/'+args.dataset)
#    train('../synth/snapshot_1e-4/synth_iter_800000.caffemodel', args.gpu)
    train(args.initmodel, args.gpu)

    write_net(args.dataset, 768, 6)
    write_solver(1e-6, 40000, 'snapshot_1e-5/'+args.dataset)
    train('snapshot_1e-4/'+args.dataset+'_iter_160000.caffemodel', args.gpu)
    write_deploy()
