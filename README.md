# Llama-3-8B-Instruct QLoRA PEFT Method

**Downloading and using Llama3**，**Xtuner based fine-tuning was performed on the MindLLM project**.

- Llama3 model application on Huggingface
- Construction of Experimental Environment
- Download and install Xtuner
- Modify the Xtuner model configuration file
- QLoRA fine-tuning based on Xtuner was carried out on the EmoLLM project

## Model and related GitHub project to download

### Llama-3-8B-Instruct download

The "Instruct" suffix indicates that this model has been fine-tuned on a dataset of instructions and tasks, making it more suitable for applications that require following instructions or completing tasks.

```python
from modelscope import snapshot_download

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct')
print(model_dir)
```

![](https://cdn.nlark.com/yuque/0/2024/png/43035260/1713539295924-090cc0a2-11e8-47cb-85e2-2bd3c8952606.png#averageHue=%23050910&clientId=uc3565ceb-1d40-4&from=paste&id=udf3bb6a7&originHeight=399&originWidth=431&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uba96494a-41be-4549-8c03-7b494d925fb&title=)

## Experimental Environment


```bash
conda create -n llama python=3.10
```

### pytorch安装

```python
# CUDA 12.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### git clone xtuner-0.1.18.dev0

```python
git clone https://github.com/InternLM/xtuner
```

`llama3_chat` template in `/root/xtuner/xtuner/utils/templates.py`

```python
llama3_chat=dict(
    SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
            '{system}<|eot_id|>'),
    INSTRUCTION=(
        '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'),
    SUFFIX='<|eot_id|>',
    SUFFIX_AS_EOS=True,
    STOP_WORDS=['<|eot_id|>']),
```

### Install xtuner-0.1.18.dev0

```python
cd /root/xtuner/xtuner

pip install -e '.[all]'
```

![](https://cdn.nlark.com/yuque/0/2024/png/43035260/1713539296026-ab8aa256-d55c-4b52-85fc-807bc7e38ed5.png#averageHue=%23384c57&clientId=uc3565ceb-1d40-4&from=paste&id=u1be0c9be&originHeight=318&originWidth=720&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u6c2f10b3-fd9b-4eed-a123-e718d30f87f&title=)



### Adjust the parameters according to the hardware configuration


![](https://cdn.nlark.com/yuque/0/2024/png/43035260/1713539296391-a1dd32ce-7dd9-4ccc-9d85-6d299dad2bf8.png#averageHue=%23080d14&clientId=uc3565ceb-1d40-4&from=paste&id=u97f1acad&originHeight=381&originWidth=652&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u9fcd4a33-10af-47f5-8031-e30c1844c49&title=)

**Set the environment variable**`PYTORCH_CUDA_ALLOC_CON`，To make better use of all the **GPU memory **

```python
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'
```

```python
max_length = 2048
batch_size = 16 # per_device
accumulative_counts = 1
max_epochs = 3
lr = 1e-4
evaluation_freq = 500
```

```python
r=32,
lora_alpha=64,
```

### Add a Chinese answer prompt

What's more, **Llama3 currently doesn't support Chinese ** very well, so you may need to add some adjustments to the 'system prompt' related to tasks such as' Chinese field 'and', and then you will only answer and ask questions in Chinese '.

```python
SYSTEM = "你由EmoLLM团队打造的中文领域心理健康助手, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验，接下来你将只使用中文来回答和咨询问题。"
```



## Data format

```python
[
    # 多轮对话
    {
        "conversation": [
            {
                "input": "我最近很迷茫，高三后的压力让我感到恐惧，我对自己的存在感到怀疑，您能不能给我一些建议呢？",
                "output": "看到你面临的困境与压力，真心希望能够帮助你。首先，我想说的是，你并不是孤独的，高考前的困惑与迷茫是很常见的。不要给自己太大的压力，给自己短暂的放松。你认识到问题所在，这已经是一个很大的进步了。",
                "system": "你是心理健康助手EmoLLM，由EmoLLM团队打造。你旨在通过专业心理咨询，协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术，一步步帮助来访者解决心理问题。"
            },
            {
                "input": "谢谢您的鼓励，但我觉得自己学习成绩不好，没有进步，这让我更加困惑。",
                "output": "学习成绩好坏不是解决问题的关键，专注于自己的进步更为重要。为了达到自己的目标，制定一份合理的计划，按部就班的完成每一项任务。有时候自学不如求助他人，不要害怕问老师和同学问题，以便获得更好的帮助和理解。"
            },
            {
                "input": "我认为我缺乏自信，不知道怎样扭转自己的消极情绪，您能给我一些建议吗？",
                "output": "建立自信需要时间和努力。让我们从一点一滴的进步中建立自信心。设立一些具体的目标，完成这些目标将会培养自己的自信。鼓励自己，在每天结束前，总结一下自己的成就，这将帮助你更好的看到自身的优点。"
            },
            {
                "input": "我感到很无助，不知道该如何调整自己的心态。",
                "output": "了解自己的情绪并寻找有效的调节方法是处理感到无助情况的关键。比如爱好运动、听音乐和阅读书籍等，这些活动对于您的心理健康都是有积极帮助的。此外，在处理感到无助的情况下，家庭和朋友的支持也是非常关键的。"
            },
            {
                "input": "我感觉我的思维和学习方法都很糟糕，该怎样改进呢？",
                "output": "学习方法确实很重要，它会对你的成绩和情绪有很大的影响。最有效的学习方法是利用时间来做练习，多做一些题目，练习自己的掌握能力。建立起对知识的兴趣，这需要不断的积累，不要害怕失败或挫折，它们会成为助你成长的宝贵财富。"
            },
            {
                "input": "我会按照您的建议努力，但想问一下，我该如何面对可能的失败？",
                "output": "每个人都会面对失败和挫折，不要让它们打败你。碰到失败时，重新审视自己，分析分析自己的不足，寻找方法去改进。记得勇敢地看向未来，不要让过去的失败阻碍了自己前进的脚步。"
            },
            {
                "input": "非常感谢您的建议和支持，我会努力去实践。",
                "output": "这是我应该做的，你有勇气面对自己困难，你并不孤独，我们会一起为你打气，祝愿你早日获得心灵的平静和所有的成功。"
            }
        ]
    },
    ......
]
```


### Fine tuning instructions

```python
cd xtuner_config
xtuner train llama3_8b_instruct_qlora_alpaca_e3_M.py --deepspeed deepspeed_zero2
```

### 训练开始的截图

![](https://cdn.nlark.com/yuque/0/2024/png/43035260/1713539297483-535d17f4-8f56-409e-8bd9-b378cde27571.png#averageHue=%230c121b&clientId=uc3565ceb-1d40-4&from=paste&id=u8dad52e2&originHeight=167&originWidth=1235&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ub8de5823-f3ef-4828-8e42-4ed425f15ae&title=)

### 500step print

![](https://cdn.nlark.com/yuque/0/2024/png/43035260/1713539298145-001f7e4d-ff2a-480a-8343-479dec79ae0a.png#averageHue=%230d131a&clientId=uc3565ceb-1d40-4&from=paste&id=ud6b548b1&originHeight=426&originWidth=720&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u2d75814e-53d3-4abb-b01a-7479a26ebb1&title=)

## The resulting PTH model is transformed into a HuggingFace model

That is, generate a HuggingFace Adapter folder to merge with the original model weights
```python
cd xtuner_config
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert pth_to_hf llama3_8b_instruct_qlora_alpaca_e3_M.py ./work_dirs/llama3_8b_instruct_qlora_alpaca_e3_M/epoch_1.pth ./hf_llama3
```

## Incorporation of HuggingFace Adapter QLoRA weights into the large language model

```python
xtuner convert merge /root/models/LLM-Research/Meta-Llama-3-8B-Instruct ./hf_llama3 ./merged_Llama3_8b_instruct --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

## Test

MindLLM_test.ipynb

## Evaluation

evaluate.ipynb


