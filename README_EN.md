<div align="center">
<h1>
  XVERSE-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-13B">🤗 HuggingFace</a>&nbsp ｜ &nbsp<a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <b>English</b> |
        <a href="README.md">中文</a>
    <p>
</h4>

## Model Introduction

**XVERSE-13B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology. Its key features are as follows:

- **Model Structure**: XVERSE-13B uses the mainstream Decoder-only Transformer network structure, supports 8k context length, the longest one among models of the same size, which can meet the need of longer multi-round dialogues, knowledge question-answering, and summarization. This makes the model more versatile in application scenarios.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 1.4 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages.
- **Tokenization**: Based on the BPE (Byte-Pair Encoding) algorithm, a tokenizer with a vocabulary size of 100,278 has been trained using hundreds of gigabytes of language data. This tokenizer is capable of supporting multilingual without the need for additional vocabulary expansion.
- **Training Framework**: Several key technologies have also been independently developed, including efficient operators, memory optimization, parallel scheduling strategies, overlap of data-computation-communication, and synergy between platforms and frameworks. These advancements enhance training efficiency and model stability. With these technologies, the peak computational power utilization rate on a thousand-card cluster can reach 58.5%, ranking at the forefront of the industry.

## Model Evaluation

In order to validate the various abilities of the model, we have chosen several comprehensive capability benchmarks across multiple disciplines, including [MMLU](https://arxiv.org/abs/2009.03300) (English), [C-Eval](https://cevalbenchmark.com/) (Chinese), [AGIEval](https://arxiv.org/abs/2304.06364) (Chinese and English), [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) (Chinese and English), [GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao) (English), the evaluation results are as follows:

|      Models\Datasets       |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
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

> <sup>1: Tests are conducted only on single-answer multiple-choice questions, thus excluding fill-in-the-blanks, open-ended questions, and multiple-answer multiple-choice questions.</sup>   
> <sup>2: Reporting results from [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B).</sup>   
> <sup>3: Reporting results from [C-Eval](https://cevalbenchmark.com/).</sup>   
> <sup>4: Reporting results from [Llama 2](https://arxiv.org/abs/2307.09288).</sup>
>
> For MMLU, we adopt the [evaluation tools](https://github.com/hendrycks/test) provided by the authors, C-Eval, AGIEval, GAOKAO-Bench, GAOKAO-English are the same as MMLU, and uniformly use **5-shot** to construct the test samples.

### MMLU Category Results
|     Models\Categories      | Average  |   STEM   | Social Science | Humanities |  Others  |
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

### C-Eval Category Results
|     Models\Categories      | Average  |   STEM   | Social Science | Humanities |  Others  |
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

## Usage

### Environment Setup

1. Clone this repository:

```shell
git clone https://github.com/xverse-ai/XVERSE-13B
cd XVERSE-13B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Loading with Transformers

The XVERSE-13B model can be loaded for inference using the following code:

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

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-13B model:

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-13B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-13B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-13B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-13B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-13B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-13B model weights are **fully open** to academic research and support **free commercial use**. Commercial use requires an application for a commercial use license by sending an email to <opensource@xverse.cn>.

