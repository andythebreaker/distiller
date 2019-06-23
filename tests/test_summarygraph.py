#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import torch
import pytest
import distiller
from distiller.models import ALL_MODEL_NAMES, create_model
from distiller.apputils import *
from distiller import normalize_module_name, denormalize_module_name, \
    SummaryGraph, onnx_name_2_pytorch_name
from distiller.model_summaries import connectivity_summary, connectivity_summary_verbose


# Logging configuration
logging.basicConfig(level=logging.DEBUG)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def create_graph(dataset, arch):
    dummy_input = distiller.get_dummy_input(dataset)
    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None
    return SummaryGraph(model, dummy_input)


def test_graph():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None


def test_connectivity():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    op_names = [op['name'] for op in g.ops.values()]
    assert len(op_names) == 80

    edges = g.edges
    assert edges[0].src == '0' and edges[0].dst == 'conv1'

    # Test two sequential calls to predecessors (this was a bug once)
    preds = g.predecessors(g.find_op('bn1'), 1)
    preds = g.predecessors(g.find_op('bn1'), 1)
    assert preds == ['129', '2', '3', '4', '5']
    # Test successors
    succs = g.successors(g.find_op('bn1'), 2)
    assert succs == ['relu']

    op = g.find_op('layer1.0.relu2')
    assert op is not None
    succs = g.successors(op, 4)
    assert succs == ['layer1.1.bn1', 'layer1.1.relu2']

    preds = g.predecessors(g.find_op('bn1'), 10)
    assert preds == []
    preds = g.predecessors(g.find_op('bn1'), 3)
    assert preds == ['0', '1']


def test_layer_search():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    op = g.find_op('layer1.0.conv1')
    assert op is not None

    succs = g.successors_f('layer1.0.conv1', 'Conv', [], logging)
    assert ['layer1.0.conv2'] == succs

    succs = g.successors_f('relu', 'Conv', [], logging)
    assert succs == ['layer1.0.conv1', 'layer1.1.conv1', 'layer1.2.conv1', 'layer2.0.conv1', 'layer2.0.downsample.0']

    succs = g.successors_f('relu', 'Gemm', [], logging)
    assert succs == ['fc']

    succs = g.successors_f('layer3.2', 'Conv', [], logging)
    assert succs == []
    #logging.debug(succs)

    preds = g.predecessors_f('conv1', 'Conv', [], logging)
    assert preds == []

    preds = g.predecessors_f('layer1.0.conv2', 'Conv', [], logging)
    assert preds == ['layer1.0.conv1']

    preds = g.predecessors_f('layer1.0.conv1', 'Conv', [], logging)
    assert preds == ['conv1']

    preds = g.predecessors_f('layer1.1.conv1', 'Conv', [], logging)
    assert preds == ['layer1.0.conv2', 'conv1']


def test_vgg():
    g = create_graph('imagenet', 'vgg19')
    assert g is not None
    succs = g.successors_f('features.32', 'Conv')
    logging.debug(succs)
    succs = g.successors_f('features.34', 'Conv')


def test_simplenet():
    g = create_graph('cifar10', 'simplenet_cifar')
    assert g is not None
    preds = g.predecessors_f(normalize_module_name('module.conv1'), 'Conv')
    logging.debug("[simplenet_cifar]: preds of module.conv1 = {}".format(preds))
    assert len(preds) == 0

    preds = g.predecessors_f(normalize_module_name('module.conv2'), 'Conv')
    logging.debug("[simplenet_cifar]: preds of module.conv2 = {}".format(preds))
    assert len(preds) == 1


def name_test(dataset, arch):
    model = create_model(False, dataset, arch, parallel=False)
    modelp = create_model(False, dataset, arch, parallel=True)
    assert model is not None and modelp is not None

    mod_names   = [mod_name for mod_name, _ in model.named_modules()]
    mod_names_p = [mod_name for mod_name, _ in modelp.named_modules()]
    assert mod_names is not None and mod_names_p is not None
    assert len(mod_names)+1 == len(mod_names_p)

    for i in range(len(mod_names)-1):
        assert mod_names[i+1] == normalize_module_name(mod_names_p[i+2])
        logging.debug("{} {} {}".format(mod_names_p[i+2], mod_names[i+1], normalize_module_name(mod_names_p[i+2])))
        assert mod_names_p[i+2] == denormalize_module_name(modelp, mod_names[i+1])


def test_normalize_module_name():
    assert "features.0" == normalize_module_name("features.module.0")
    assert "features.0" == normalize_module_name("module.features.0")
    assert "features" == normalize_module_name("features.module")
    name_test('imagenet', 'vgg19')
    name_test('cifar10', 'resnet20_cifar')
    name_test('imagenet', 'alexnet')


def named_params_layers_test_aux(dataset, arch, dataparallel:bool):
    model = create_model(False, dataset, arch, parallel=dataparallel)
    sgraph = SummaryGraph(model, distiller.get_dummy_input(dataset))
    sgraph_layer_names = set(k for k, i, j in sgraph.named_params_layers())
    for layer_name in sgraph_layer_names:
        assert sgraph.find_op(layer_name) is not None, '{} was not found in summary graph'.format(layer_name)


def test_named_params_layers():
    for dataParallelModel in (True, False):
        named_params_layers_test_aux('imagenet', 'vgg19', dataParallelModel)
        named_params_layers_test_aux('cifar10', 'resnet20_cifar', dataParallelModel)
        named_params_layers_test_aux('imagenet', 'alexnet', dataParallelModel)
        named_params_layers_test_aux('imagenet', 'resnext101_32x4d', dataParallelModel)


def test_onnx_name_2_pytorch_name():
    assert onnx_name_2_pytorch_name("ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]") == "layer3.0.relu"
    assert onnx_name_2_pytorch_name('VGG/[features]/Sequential/Conv2d[34]') == "features.34"
    assert onnx_name_2_pytorch_name('NameWithNoModule') == 


def test_connectivity_summary():
    g = create_graph('cifar10', 'resnet20_cifar')
    assert g is not None

    summary = connectivity_summary(g)
    assert len(summary) == 80

    verbose_summary = connectivity_summary_verbose(g)
    assert len(verbose_summary) == 80


def test_sg_macs():
    'Compare the MACs of different modules as computed by a SummaryGraph
    and model summary.'
    import common
    sg = create_graph('imagenet', 'mobilenet')
    assert sg
    model, _ = common.setup_test('mobilenet', 'imagenet', parallel=False)
    df_compute = distiller.model_performance_summary(model, distiller.get_dummy_input('imagenet'))
    modules_macs = df_compute.loc[:, ['Name', 'MACs']]
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            summary_macs = int(modules_macs.loc[modules_macs.Name == name].MACs)
            sg_macs = sg.find_op(name)['attrs']['MACs']
            assert summary_macs == sg_macs
 

def test_weights_size_attr():
    def test(dataset, arch, dataparallel:bool):
        model = create_model(False, dataset, arch, parallel=dataparallel)
        sgraph = SummaryGraph(model, distiller.get_dummy_input(dataset))

        distiller.assign_layer_fq_names(model)
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                op = sgraph.find_op(name)
                assert op is not None
                assert op['attrs']['weights_vol'] == distiller.volume(mod.weight)

    for data_parallel in (True, False):
        test('cifar10', 'resnet20_cifar', data_parallel)
        test('imagenet', 'alexnet', data_parallel)
        test('imagenet', 'resnext101_32x4d', data_parallel)


if __name__ == '__main__':
    #test_connectivity_summary()
    test_sg_macs()