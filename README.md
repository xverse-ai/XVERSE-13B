<div align="center">
<h1>
  XVERSE-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-13B">🤗 HuggingFace</a>&nbsp ｜ &nbsp<a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 模型介绍

**XVERSE-13B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），主要特点如下：

- **模型结构**：XVERSE-13B 使用主流 Decoder-only 的标准 Transformer 网络结构，支持 8K 的上下文长度（Context Length），为同尺寸模型中最长，能满足更长的多轮对话、知识问答与摘要等需求，模型应用场景更广泛。
- **训练数据**：构建了 1.4 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果。
- **分词**：基于 BPE（Byte-Pair Encoding）算法，使用上百 GB 语料训练了一个词表大小为 100,278 的分词器，能够同时支持多语言，而无需额外扩展词表。
- **训练框架**：自主研发多项关键技术，包括高效算子、显存优化、并行调度策略、数据-计算-通信重叠、平台和框架协同等，让训练效率更高，模型稳定性强，在千卡集群上的峰值算力利用率可达到 58.5%，位居业界前列。

## 评测结果

为验证模型的各项能力，我们选取了多个学科综合能力评测集，包括 [MMLU](https://arxiv.org/abs/2009.03300)（英文）、 [C-Eval](https://cevalbenchmark.com/)（中文）、[AGIEval](https://arxiv.org/abs/2304.06364)（中英） 、[GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)（中英）、[GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao)（英文），评测结果如下：

|        模型\数据集         |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
| :------------------------: | :--------------: | :--------------: | :-----------------: | :----------------------: | :------------------------: |
|        Baichuan-13B        | 51.6<sup>2</sup> | 53.6<sup>3</sup> |        40.5         |           45.9           |            56.9            |
|        Llama-1-13B         | 46.9<sup>4</sup> |       28.8       |        27.3         |           26.4           |            38.1            |
|        Llama-2-13B         | 54.8<sup>4</sup> |       35.6       |        33.4         |           35.4           |            60.6            |
|  moss-moon-003-base (16B)  |       24.7       | 33.1<sup>3</sup> |        26.8         |           28.5           |            34.7            |
|       OpenLLaMA-13B        |       42.4       |       24.7       |        24.0         |           25.6           |            33.3            |
|          OPT-13B           |       25.2       |       25.0       |        24.2         |           24.4           |            31.1            |
|         Pythia-12B         |       25.1       |       26.2       |        25.3         |           25.3           |            26.8            |
| Ziya-LLaMA-13B-Pretrain-v1 |       43.9       |       30.2       |        27.2         |           26.4           |            37.6            |
|       **XVERSE-13B**       |     **55.1**     |     **54.7**     |      **41.4**       |         **53.9**         |          **66.5**          |

> <sup>1：只针对其中的单项选择题进行测试，即排除了填空题、开放性问题和多项选择题</sup>   
> <sup>2：来源于 [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) 的汇报结果</sup>   
> <sup>3：来源于 [C-Eval](https://cevalbenchmark.com/) 的汇报结果</sup>   
> <sup>4：来源于[Llama 2 论文](https://arxiv.org/abs/2307.09288)的汇报结果</sup>
>
> 对于 MMLU ，我们采用作者提供的[评测工具](https://github.com/hendrycks/test)，C-Eval、AGIEval、GAOKAO-Bench、GAOKAO-English 与 MMLU 的评测方式相同，且统一采用 **5-shot** 构造测试样本。

### MMLU 各类别指标
|         模型\类别          | Average  |   STEM   | Social Science | Humanities |  Others  |
| :------------------------: | :------: | :------: | :------------: | :--------: | :------: |
|        Baichuan-13B        |   51.6   |   41.6   |      60.9      |    47.4    |   58.5   |
|        Llama-1-13B         |   46.9   |   35.8   |      53.8      |    45.0    |   53.3   |
|        Llama-2-13B         |   54.8   |   44.1   |      62.6      |    52.8    |   61.1   |
|  moss-moon-003-base (16B)  |   24.7   |   23.0   |      24.0      |    25.2    |   26.3   |
|       OpenLLaMA-13B        |   42.4   |   34.7   |      48.6      |    40.0    |   47.1   |
|          OPT-13B           |   25.2   |   23.9   |      24.1      |    25.9    |   26.3   |
|         Pythia-12B         |   25.1   |   24.8   |      23.0      |    26.1    |   26.0   |
| Ziya-LLaMA-13B-Pretrain-v1 |   43.9   |   36.3   |      48.8      |    41.1    |   50.3   |
|       **XVERSE-13B**       | **55.1** | **44.5** |    **64.4**    |  **50.5**  | **62.9** |

### C-Eval 各类别指标
|         模型\类别          | Average  |   STEM   | Social Science | Humanities |  Others  |
| :------------------------: | :------: | :------: | :------------: | :--------: | :------: |
|        Baichuan-13B        |   53.6   |   47.0   |      66.8      |    57.3    |   49.8   |
|        Llama-1-13B         |   28.8   |   27.5   |      33.9      |    27.7    |   27.7   |
|        Llama-2-13B         |   35.6   |   34.5   |      39.8      |    36.2    |   33.2   |
|  moss-moon-003-base (16B)  |   33.1   |   31.6   |      37.0      |    33.4    |   32.1   |
|       OpenLLaMA-13B        |   24.7   |   25.5   |      23.5      |    24.2    |   24.7   |
|          OPT-13B           |   25.0   |   24.4   |      24.6      |    25.9    |   25.4   |
|         Pythia-12B         |   26.2   |   26.8   |      25.1      |    26.7    |   25.4   |
| Ziya-LLaMA-13B-Pretrain-v1 |   30.2   |   27.8   |      34.3      |    32.0    |   29.0   |
|       **XVERSE-13B**       | **54.7** | **45.6** |    **66.2**    |  **58.3**  | **56.9** |

## 使用方法

### 环境安装

1. 下载本仓库：

```shell
git clone https://github.com/xverse-ai/XVERSE-13B
cd XVERSE-13B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```
### Transformers 加载方式

可通过以下代码加载 XVERSE-13B 模型进行推理：

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-13B")
>>> model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-13B", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
>>> inputs = inputs.cuda()
>>> generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
>>> print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

### 网页 Demo

可通过以下代码启动一个web server，在浏览器输入访问地址后，可使用 XVERSE-13B 模型进行推理：

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## 局限性与免责申明

XVERSE-13B 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-13B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-13B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-13B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-13B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-13B 模型权重对学术研究**完全开放**，并且支持**免费商用**，商用需申请商业使用授权，可以发送邮件到 <opensource@xverse.cn> 进行申请。

