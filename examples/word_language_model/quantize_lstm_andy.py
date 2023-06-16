# %%
"""
# Quantizing RNN Models
"""

# %%
import sys
sys.path.append("../..")

# %%
"""
In this example, we show how to quantize recurrent models.  
Using a pretrained model `model.RNNModel`, we convert the built-in pytorch implementation of LSTM to our own, modular implementation.  
The pretrained model was generated with:  
```time python3 main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --tied --wd=1e-6```  
The reason we replace the LSTM that is because the inner operations in the pytorch implementation are not accessible to us, but we still want to quantize these operations. <br />
Afterwards we can try different techniques to quantize the whole model.  
"""

# %%
from model import DistillerRNNModel, RNNModel
from data import Corpus
import torch
from torch import nn
import distiller
from distiller.modules import DistillerLSTM as LSTM
from tqdm import tqdm # for pretty progress bar
import numpy as np
from copy import deepcopy

import warnings
warnings.filterwarnings(action='default', module='distiller.quantization')
warnings.filterwarnings(action='default', module='distiller.quantization.range_linear')

# %%
"""
### Preprocess the data
"""

# %%
corpus = Corpus('./data/wikitext-2/')

# %%
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
device = 'cuda:0'
batch_size = 20
eval_batch_size = 10
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# %%
"""
### Loading the model and converting to our own implementation
"""

# %%
rnn_model = torch.load('./checkpoint.pth.tar.best')
rnn_model = rnn_model.to(device)
rnn_model

# %%
"""
Here we convert the pytorch LSTM implementation to our own, by calling `LSTM.from_pytorch_impl`:
"""

# %%
def manual_model(pytorch_model_: RNNModel):
    nlayers, ninp, nhid, ntoken, tie_weights = \
        pytorch_model_.nlayers, \
        pytorch_model_.ninp, \
        pytorch_model_.nhid, \
        pytorch_model_.ntoken, \
        pytorch_model_.tie_weights

    model = DistillerRNNModel(nlayers=nlayers, ninp=ninp, nhid=nhid, ntoken=ntoken, tie_weights=tie_weights).to(device)
    model.eval()
    model.encoder.weight = nn.Parameter(pytorch_model_.encoder.weight.clone().detach())
    model.decoder.weight = nn.Parameter(pytorch_model_.decoder.weight.clone().detach())
    model.decoder.bias = nn.Parameter(pytorch_model_.decoder.bias.clone().detach())
    model.rnn = LSTM.from_pytorch_impl(pytorch_model_.rnn)

    return model

man_model = manual_model(rnn_model)
torch.save(man_model, 'manual.checkpoint.pth.tar')
man_model

# %%
"""
### Batching the data for evaluation
"""

# %%
sequence_len = 35
def get_batch(source, i):
    seq_len = min(sequence_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

hidden = rnn_model.init_hidden(eval_batch_size)
data, targets = get_batch(test_data, 0)

# %%
"""
### Check that the convertion has succeeded
"""

# %%
rnn_model.eval()
man_model.eval()
y_t, h_t = rnn_model(data, hidden)
y_p, h_p = man_model(data, hidden)

print("Max error in y: %f" % (y_t-y_p).abs().max().item())

# %%
"""
### Defining the evaluation
"""

# %%
criterion = nn.CrossEntropyLoss()
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    

def evaluate(model, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        with tqdm(range(0, data_source.size(0), sequence_len)) as t:
            # The line below was fixed as per: https://github.com/pytorch/examples/issues/214
            for i in t:
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
                avg_loss = total_loss / (i + 1)
                t.set_postfix((('val_loss', avg_loss), ('ppl', np.exp(avg_loss))))
    return total_loss / len(data_source)

# %%
"""
# Quantizing the Model
"""

# %%
"""
## Collect activation statistics
"""

# %%
"""
The model uses activation statistics to determine how big the quantization range is. The bigger the range - the larger the round off error after quantization which leads to accuracy drop.  
Our goal is to minimize the range s.t. it contains the absolute most of our data.  
After that, we divide the range into chunks of equal size, according to the number of bits, and transform the data according to this scale factor.  
Read more on scale factor calculation [in our docs](https://intellabs.github.io/distiller/algo_quantization.html).

The class `QuantCalibrationStatsCollector` collects the statistics for defining the range $r = max - min$.  

Each forward pass, the collector records the values of inputs and outputs, for each layer:
- absolute over all batches min, max (stored in `min`, `max`)
- average over batches, per batch min, max (stored in `avg_min`, `avg_max`)
- mean
- std
- shape of output tensor  

All these values can be used to define the range of quantization, e.g. we can use the absolute `min`, `max` to define the range.
"""

# %%
import os
from distiller.data_loggers import collect_quant_stats, QuantCalibrationStatsCollector

man_model = torch.load('./manual.checkpoint.pth.tar')
distiller.utils.assign_layer_fq_names(man_model)
collector = QuantCalibrationStatsCollector(man_model)

stats_file = './acts_quantization_stats.yaml'

if not os.path.isfile(stats_file):
    def eval_for_stats(model):
        evaluate(man_model, val_data)
    collect_quant_stats(man_model, eval_for_stats, save_dir='.')

# %%
"""
## Prepare the Model For Quantization
  
We quantize the model after the training has completed.  
Here we check the baseline model perplexity, to have an idea how good the quantization is.
"""

# %%
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy

# Load and evaluate the baseline model.
man_model = torch.load('./manual.checkpoint.pth.tar')
val_loss = evaluate(man_model, val_data)
print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
Now we do our magic - __Preparing the model for quantization__.  
The quantizer replaces the layers in out model with their quantized versions.  
"""

# %%
# Define the quantizer
quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats=stats_file)

# Quantizer magic
stats_before_prepare = deepcopy(quantizer.model_activation_stats)
dummy_input = (torch.zeros(1,1).to(dtype=torch.long), man_model.init_hidden(1))
quantizer.prepare_model(dummy_input)

# %%
"""
### Net-Aware Quantization

Note that we passes a dummy input to `prepare_model`. This is required for the quantizer to be able to create a graph representation of the model, and to infer the connectivity between the modules.  
Understanding the connectivity of the model is required to enable **"Net-aware quantization"**. This term (coined in [\[1\]](#references), section 3.2.2), means we can achieve better quantization by considering sequences of operations.  
In the case of LSTM, we have an element-wise add operation whose output is split into 4 and fed into either Tanh or Sigmoid activations. Both of these ops saturate at relatively small input values - tanh at approximately $|4|$, and sigmoid saturates at approximately $|6|$. This means we can safely clip the output of the element-wise add operation between $[-6,6]$. `PostTrainLinearQuantizer` detects this patterm and modifies the statistics accordingly.
"""

# %%
import pprint
pp = pprint.PrettyPrinter(indent=1)
print('Stats BEFORE prepare_model:')
pp.pprint(stats_before_prepare['rnn.cells.0.eltwiseadd_gate']['output'])

print('\nStats AFTER to prepare_model:')
pp.pprint(quantizer.model_activation_stats['rnn.cells.0.eltwiseadd_gate']['output'])

# %%
"""
Note the value for `avg_max` did not change, since it was already below the clipping value of $6.0$.
"""

# %%
"""
### Inspecting the Quantized Model

Let's see how the model has after being prepared for quantization:
"""

# %%
quantizer.model

# %%
"""
Note how `encoder` and `decoder` have been replaced with wrapper layers (for the relevant module type), which handle the quantization. The same holds for the internal layers of the `DistillerLSTM` module, which we don't print for brevity sake. To "peek" inside the `DistillerLSTM` module, we need to access it directly. As an example, let's take a look at a couple of the internal layers:
"""

# %%
print(quantizer.model.rnn.cells[0].fc_gate_x)
print(quantizer.model.rnn.cells[0].eltwiseadd_gate)

# %%
"""
## Running the Quantized Model

### Try 1: Initial settings - simple symmetric quantization

Finally, let's go ahead and evaluate the quantized model:
"""

# %%
val_loss = evaluate(quantizer.model.to(device), val_data)
print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
As we can see here, the perplexity has increased much - meaning our quantization has damaged the accuracy of our model.
"""

# %%
"""
### Try 2: Assymetric, per-channel

Let's try quantizing each channel separately, and making the range of the quantization asymmetric.
"""

# %%
quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats=stats_file,
    mode=LinearQuantMode.ASYMMETRIC_SIGNED,
    per_channel_wts=True
)
quantizer.prepare_model(dummy_input)
quantizer.model

# %%
val_loss = evaluate(quantizer.model.to(device), val_data)
print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
A tiny bit better, but still no good.
"""

# %%
"""
### Try 3: Mixed FP16 and INT8

Let us try the half precision (aka FP16) version of the model:
"""

# %%
model_fp16 = deepcopy(man_model).half()
val_loss = evaluate(model_fp16, val_data)
print('val_loss: %8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
The result is very close to our original model! That means that the roundoff when quantizing linearly to 8-bit integers is what hurts our accuracy.

Luckily, `PostTrainLinearQuantizer` supports quantizing some/all layers to FP16 using the `fp16` parameter. In light of what we just saw, and as stated in [\[2\]](#References), let's try keeping element-wise operations at FP16, and quantize everything else to 8-bit using the same settings as in try 2.
"""

# %%
overrides_yaml = """
.*eltwise.*:
    fp16: true
encoder:
    fp16: true
decoder:
    fp16: true
"""
overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats=stats_file,
    mode=LinearQuantMode.ASYMMETRIC_SIGNED,
    overrides=overrides,
    per_channel_wts=True
)
quantizer.prepare_model(dummy_input)
quantizer.model

# %%
val_loss = evaluate(quantizer.model.to(device), val_data)
print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
The accuracy is still holding up very well, even though we quantized the inner linear layers!  
"""

# %%
"""
### Try 4: Clipping Activations

Now, lets try to choose different boundaries for `min`, `max`.  
Instead of using absolute ones, we take the average of all batches (`avg_min`, `avg_max`), which is an indication of where usually most of the boundaries lie. This is done by specifying the `clip_acts` parameter to `ClipMode.AVG` or `"AVG"` in the quantizer ctor:
"""

# %%
overrides_yaml = """
encoder:
    fp16: true
decoder:
    fp16: true
"""
overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats=stats_file,
    mode=LinearQuantMode.ASYMMETRIC_SIGNED,
    overrides=overrides,
    per_channel_wts=True,
    clip_acts="AVG"
)
quantizer.prepare_model(dummy_input)
val_loss = evaluate(quantizer.model.to(device), val_data)
print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
"""
Great! Even though we quantized all of the layers except the embedding and the decoder - we got almost no accuracy penalty. Lets try quantizing them as well:
"""

# %%
quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats=stats_file,
    mode=LinearQuantMode.ASYMMETRIC_SIGNED,
    per_channel_wts=True,
    clip_acts="AVG"
)
quantizer.prepare_model(dummy_input)
val_loss = evaluate(quantizer.model.to(device), val_data)
print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))

# %%
quantizer.model

# %%
"""
Here we see that sometimes quantizing with the right boundaries gives better results than actually using floating point operations (even though they are half precision). 
"""

# %%
"""
## Conclusion

Choosing the right boundaries for quantization  was crucial for achieving almost no degradation in accrucay of LSTM.  
  
Here we showed how to use the Distiller quantization API to quantize an RNN model, by converting the PyTorch implementation into a modular one and then quantizing each layer separately.
"""

# %%
"""
## References

1. **Jongsoo Park, Maxim Naumov, Protonu Basu, Summer Deng, Aravind Kalaiah, Daya Khudia, James Law, Parth Malani, Andrey Malevich, Satish Nadathur, Juan Miguel Pino, Martin Schatz, Alexander Sidorov, Viswanath Sivakumar, Andrew Tulloch, Xiaodong Wang, Yiming Wu, Hector Yuen, Utku Diril, Dmytro Dzhulgakov, Kim Hazelwood, Bill Jia, Yangqing Jia, Lin Qiao, Vijay Rao, Nadav Rotem, Sungjoo Yoo, Mikhail Smelyanskiy**. Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications. [arxiv:1811.09886](https://arxiv.org/abs/1811.09886)

2. **Qinyao He, He Wen, Shuchang Zhou, Yuxin Wu, Cong Yao, Xinyu Zhou, Yuheng Zou**. Effective Quantization Methods for Recurrent Neural Networks. [arxiv:1611.10176](https://arxiv.org/abs/1611.10176)
"""