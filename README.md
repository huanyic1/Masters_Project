# C-Sharp

*Codesign for Sparsified Hardware-Accelerated Reuse Propagation in Transformers*,
or C-Sharp for short,
is a reapplication of the gradient-reuse ideas originally proposed in the [ReSprop paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Goli_ReSprop_Reuse_Sparsified_Backpropagation_CVPR_2020_paper.pdf) to the training of transformer architectures.


## Usage
The `ReSpropLinear` layer works just like a regular `Linear` torch layer,
with the addition of the `att_reuse_schedule` and `lin_reuse_schedule` parameter to specify the gradient reuse percentage / sparsity.

For Bert models, we also support automatic respropification using the `respropify_bert_att_k` and `patch_bert_self_attention_k` functions:

```python
from resprop_attention_k import respropify_bert_att_k, patch_bert_self_attention_k
from transformers import AutoModel

model = AutoModel("bert-base-uncased")
model = respropify_bert_att_k(model, att_reuse_schedule=[[0.0,0],[0.9, 0.25]], lin_reuse_schedule=[[0.0,0],[0.9, 0.25]]) # Starts with no reuse, then switches to 90% reuse 25% of the way through training for both linear and attention layers
```

> **Attention:**
> The provided implementation at the moment only serves to demonstrate that one can reuse gradients during training, without taking advantage of the sparsity.
> Consequently, backpropagation is actually slower when using it.


## Run Experiments
We recommend you use [pixi](http://pixi.sh) to set up the environment and run the experiment.

```bash
pixi install
pixi run python ...
```

Alternatively, you can also set up your environment manually.

```bash
pip install torch==2.5.1 transformers==4.46.3 accelerate==1.1.1 scikit-learn==1.5.2 matplotlib==3.9.2 datasets==3.1.0 evaluate==0.4.3
python ...
```

Download and preprocess the wiki and openweb dataset with preprocess_openweb_wiki.py

```bash
pixi run python preprocess_openweb_wiki.py --out_dir ./cached_data --num_proc NUM_GPUS
```
Pretrain from scratch with `bert_runner.py`

```bash
pixi run torchrun  --nnodes 1  --nproc_per_node 4 bert_runner.py  --model_path ./semi_trained_bert  --output_dir ./large_att_6_8 --cache_dir ./training_data/  --batch
_size 32 --num_accum 2  --epochs 3  --mlm_probability 0.15   --num_proc 4 --max_steps 100000  --plot_name large_att_6_8  
```


Depending on the reuse schedule specified in bert_runner.py, this can take anywhere from 18-24 hours for 100000 samples,
at the end of which a file called `{plot_name}.png` should be generated,
showing training loss. 
