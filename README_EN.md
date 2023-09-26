<div align="center">
<h1>
  XVERSE-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-13B">🤗 XVERSE-13B</a>&nbsp｜&nbsp<a href="https://huggingface.co/xverse/XVERSE-13B-Chat">🤗 XVERSE-13B-Chat</a>&nbsp｜&nbsp
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜&nbsp
        <a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b> |
        <a href="README_JA.md">日本語</a>
    <p>
</h4>

## Update Information
**[2023/09/26]** Released the [XVERSE-7B](https://github.com/xverse-ai/XVERSE-7B) base model and [XVERSE-7B-Chat](https://github.com/xverse-ai/XVERSE-7B) instruct-finetuned model with 7B size, which support deployment and operation on a single consumer-grade graphics card while maintaining high performance, full open source, and free for commercial use.   
**[2023/08/22]** Released the aligned instruct-finetuned model XVERSE-13B-Chat.

## Model Introduction

**XVERSE-13B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology. Its key features are as follows:

- **Model Structure**: XVERSE-13B uses the mainstream Decoder-only Transformer network structure, supports 8k context length, the longest one among models of the same size, which can meet the need of longer multi-round dialogues, knowledge question-answering, and summarization. This makes the model more versatile in application scenarios.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 1.4 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages.
- **Tokenization**: Based on the BPE (Byte-Pair Encoding) algorithm, a tokenizer with a vocabulary size of 100,278 has been trained using hundreds of gigabytes of language data. This tokenizer is capable of supporting multilingual without the need for additional vocabulary expansion.
- **Training Framework**: Several key technologies have also been independently developed, including efficient operators, memory optimization, parallel scheduling strategies, overlap of data-computation-communication, and synergy between platforms and frameworks. These advancements enhance training efficiency and model stability. With these technologies, the peak computational power utilization rate on a thousand-card cluster can reach 58.5%, ranking at the forefront of the industry.

## Model Evaluation

In order to validate the various abilities of the model, we have chosen several comprehensive capability benchmarks across multiple disciplines, including [MMLU](https://arxiv.org/abs/2009.03300) (English), [C-Eval](https://cevalbenchmark.com/) (Chinese), [AGIEval](https://arxiv.org/abs/2304.06364) (Chinese and English), [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) (Chinese and English), [GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao) (English), the evaluation results are as follows:


|        Models         |       Type       |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
| :------------------------: | :--------------: | :--------------: | :--------------: | :-----------------: | :----------------------: | :------------------------: |
|        Baichuan-13B       |      pretrained       | 51.6<sup>2</sup> | 53.6<sup>3</sup> |        40.5         |           45.9           |            56.9            |
|     Baichuan-13B-Chat     |     fine-tuned        | 52.1<sup>2</sup> | 51.5<sup>2</sup> |        34.6         |           46.7           |            63.8            |
|    Chinese-Alpaca-2-13B   |     fine-tuned        |       53.2       |       41.3       |        36.6         |           38.4           |            65.1            |
|        Llama-1-13B        |     pretrained        | 46.9<sup>4</sup> |       28.8       |        27.3         |           26.4           |            38.1            |
|        Llama-2-13B        |     pretrained        | 54.8<sup>4</sup> |       35.6       |        33.4         |           35.4           |            60.6            |
|  moss-moon-003-base (16B) |     pretrained        |       24.7       | 33.1<sup>3</sup> |        26.8         |           28.5           |            34.7            |
|  moss-moon-003-sft (16B)  |     fine-tuned    |       25.5       |       33.6       |        27.6         |           28.8           |            29.2            |
|       OpenLLaMA-13B       |     pretrained    |       42.4       |       24.7       |        24.0         |           25.6           |            33.3            |
|          OPT-13B          |     pretrained    |       25.2       |       25.0       |        24.2         |           24.4           |            31.1            |
|         Pythia-12B        |     pretrained    |       25.1       |       26.2       |        25.3         |           25.3           |            26.8            |
|      Vicuna-13B-v1.5      |     fine-tuned    |       53.5       |       27.9       |        29.7         |           31.6           |            52.9            |
| Ziya-LLaMA-13B-Pretrain-v1|     pretrained    |       43.9       |       30.2       |        27.2         |           26.4           |            37.6            |
|    Ziya-LLaMA-13B-v1.1    |     fine-tuned    |       50.6       |       29.3       |        23.6         |           26.7           |            27.3            |
|       **XVERSE-13B**      |     pretrained    |     **55.1**     |     **54.7**     |      **41.4**       |         **53.9**         |          **66.5**          |
|    **XVERSE-13B-Chat**    |     fine-tuned    |     **60.2**     |     **53.1**     |      **48.3**       |         **50.7**         |          **80.6**          |


> <sup>1: Tests are conducted only on single-answer multiple-choice questions, thus excluding fill-in-the-blanks, open-ended questions, and multiple-answer multiple-choice questions.</sup>   
> <sup>2: Reporting results from [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B).</sup>   
> <sup>3: Reporting results from [C-Eval](https://cevalbenchmark.com/).</sup>   
> <sup>4: Reporting results from [Llama 2](https://arxiv.org/abs/2307.09288).</sup>
>
> For MMLU, we adopt the [evaluation tools](https://github.com/hendrycks/test) provided by the authors, C-Eval, AGIEval, GAOKAO-Bench, GAOKAO-English are the same as MMLU, and uniformly use **5-shot** to construct the test samples.

### MMLU Category Results
|         Models          |         Type          | Average  |   STEM   | Social Science | Humanities |  Others  |
| :------------------------: | :------------------------: | :------: | :------: | :------------: | :--------: | :------: |
|        Baichuan-13B        |   pretrained   |   51.6   |   41.6   |      60.9      |    47.4    |   58.5   |
|     Baichuan-13B-Chat      |   fine-tuned   |   52.1   |   40.9   |      60.9      |    48.8    |   59.0   |
|    Chinese-Alpaca-2-13B    |   fine-tuned   |   53.2   |   41.8   |      61.2      |    51.3    |   59.2   |
|        Llama-1-13B         |   pretrained   |   46.9   |   35.8   |      53.8      |    45.0    |   53.3   |
|        Llama-2-13B         |   pretrained   |   54.8   |   44.1   |      62.6      |    52.8    |   61.1   |
|  moss-moon-003-base (16B)  |   pretrained   |   24.7   |   23.0   |      24.0      |    25.2    |   26.3   |
|  moss-moon-003-sft (16B)   |   fine-tuned   |   25.5   |   25.9   |      23.8      |    27.1    |   24.4   |
|       OpenLLaMA-13B        |   pretrained   |   42.4   |   34.7   |      48.6      |    40.0    |   47.1   |
|          OPT-13B           |   pretrained   |   25.2   |   23.9   |      24.1      |    25.9    |   26.3   |
|         Pythia-12B         |   pretrained   |   25.1   |   24.8   |      23.0      |    26.1    |   26.0   |
|      Vicuna-13B-v1.5       |   fine-tuned   |   53.5   |   42.3   |      61.3      |    50.3    |   60.9   |
| Ziya-LLaMA-13B-Pretrain-v1 |   pretrained   |   43.9   |   36.3   |      48.8      |    41.1    |   50.3   |
|    Ziya-LLaMA-13B-v1.1     |   fine-tuned   |   50.6   |   40.7   |      57.8      |    48.1    |   56.7   |
|       **XVERSE-13B**       |   pretrained   | **55.1** | **44.5** |    **64.4**    |  **50.5**  | **62.9** |
|    **XVERSE-13B-Chat**     |   fine-tuned   | **60.2** | **48.1** |    **67.7**    |  **56.4**  | **68.0** |

### C-Eval Category Results
|         Models          |         Type          | Average  |   STEM   | Social Science | Humanities |  Others  |
| :------------------------: | :------------------------: | :------: | :------: | :------------: | :--------: | :------: |
|        Baichuan-13B        |   pretrained  |   53.6   |   47.0   |      66.8      |    57.3    |   49.8   |
|     Baichuan-13B-Chat      |   fine-tuned  |   51.5   |   43.7   |      64.6      |    56.2    |   49.2   |
|    Chinese-Alpaca-2-13B    |   fine-tuned  |   41.3   |   37.8   |      51.1      |    42.4    |   37.8   |
|        Llama-1-13B         |   pretrained  |   28.8   |   27.5   |      33.9      |    27.7    |   27.7   |
|        Llama-2-13B         |   pretrained  |   35.6   |   34.5   |      39.8      |    36.2    |   33.2   |
|  moss-moon-003-base (16B)  |   pretrained  |   33.1   |   31.6   |      37.0      |    33.4    |   32.1   |
|  moss-moon-003-sft (16B)   |   fine-tuned  |   33.6   |   31.4   |      38.6      |    33.8    |   32.9   |
|       OpenLLaMA-13B        |   pretrained  |   24.7   |   25.5   |      23.5      |    24.2    |   24.7   |
|          OPT-13B           |   pretrained  |   25.0   |   24.4   |      24.6      |    25.9    |   25.4   |
|         Pythia-12B         |   pretrained  |   26.2   |   26.8   |      25.1      |    26.7    |   25.4   |
|      Vicuna-13B-v1.5       |   fine-tuned  |   27.9   |   25.4   |      33.2      |    29.3    |   26.2   |
| Ziya-LLaMA-13B-Pretrain-v1 |   pretrained  |   30.2   |   27.8   |      34.3      |    32.0    |   29.0   |
|    Ziya-LLaMA-13B-v1.1     |   fine-tuned  |   29.3   |   27.5   |      32.8      |    29.7    |   29.0   |
|       **XVERSE-13B**       |   pretrained  | **54.7** | **45.6** |    **66.2**    |  **58.3**  | **56.9** |
|    **XVERSE-13B-Chat**     |   fine-tuned  | **53.1** | **44.5** |    **65.3**    |  **56.5**  | **54.3** |

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

The XVERSE-13B model can be loaded to chat for inference using the following code:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> from transformers.generation.utils import GenerationConfig
>>> model_path = "xverse/XVERSE-13B-Chat"
>>> tokenizer = AutoTokenizer.from_pretrained(model_path)
>>> model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
>>> model.generation_config = GenerationConfig.from_pretrained(model_path)
>>> model = model.eval()
>>> history = [{"role": "user", "content": "1955年谁是美国总统？他是什么党派？"}]
>>> response = model.chat(tokenizer, history)
>>> print(response)
1955年,美国总统是德怀特·D·艾森豪威尔。他所属的党派是共和党。
>>> history.append({"role": "assistant", "content": response})
>>> history.append({"role": "user", "content": "他任职了多少年"})
>>> response = model.chat(tokenizer, history)
>>> print(response)
德怀特·D·艾森豪威尔在1953年至1961年间担任美国总统,所以他一共任职了8年。
```

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-13B model:

```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

### XVERSE-13B-Chat Output Example
Below are some examples of XVERSE-13B-Chat obtained using `chat_demo.py`:

<details><summary><b>Role Play</b></summary>

```
用户:
扮演杨过

XVERSE-13B-Chat:
好的,我现在要扮演的是杨过这个角色。他是金庸先生的武侠小说《神雕侠侣》中的主人公之一。现在你可以开始和我对话了。

用户:
你是谁

XVERSE-13B-Chat:
我叫杨过,是个孤儿,被郭靖夫妇收养。

用户:
你的妻子是谁

XVERSE-13B-Chat:
我的妻子是小龙女。

用户:
退出角色扮演

XVERSE-13B-Chat:
好的,下次再见!

用户:
你是谁

XVERSE-13B-Chat:
我是一个人工智能语言模型,名为XChat,由元象公司开发。我基于Transformer架构进行训练,能够理解和生成自然语言。我可以回答问题、提供信息、参与对话以及创作文本等任务。需要注意的是,我没有自我意识或个人情感,只是一个能理解和生成文本的工具。我的知识截止到2023年7月,所以关于更近期事件和发展的信息可能不包括在内。如果有任何问题,我会尽力提供帮助!
```
</details>

<details><summary><b>Quiz</b></summary>

```
用户:
《人类简史》这本书的作者是谁？请简要概括一下这本书的内容

XVERSE-13B-Chat:
《人类简史》这本书的作者是尤瓦尔·赫拉利。这是一部关于人类历史的非虚构作品,它从大约70万年前的人类起源开始,一直讲述到现代社会。

书中详细描述了人类如何从狩猎采集的生活方式过渡到农业文明,然后又发展为工业革命和科技革命。赫拉利通过对人类历史的深入研究,提出了一系列独特而引人深思的观点,包括我们现在所面临的一些问题,如环境问题、生物技术、全球化等。

他认为,人类的历史可以分为四个阶段:认知革命(约7万年前),农业革命(约1万年前),一体化革命(约500年前),以及科学革命(约400年前)。每一个阶段都带来了巨大的变革,改变了人类的生活方式,也塑造了我们的文化和社会结构。
```
</details>

<details><summary><b>Text Generation</b></summary>

```
用户:
我要去深圳旅游，帮我制定一份5日游计划

XVERSE-13B-Chat:
当然可以。以下是一份为期五天的深圳旅行建议行程:

第一天:到达深圳并休息
上午:抵达深圳后,您可以先安排入住酒店并稍作休息。然后前往深圳湾公园散步,欣赏美丽的海景。
下午:在附近的购物中心或餐厅享用午餐,之后你可以选择去欢乐谷或者东部华侨城游玩。
晚上:在世界之窗体验各种风格的建筑和文化,结束你的第一天。

第二天:深圳科技与现代化
上午:参观深圳市民中心、荔香公园以及深圳图书馆。
下午:走进深圳会展中心和深圳大剧院,感受深圳的现代化气息。
晚上:在南山区的海岸城购物娱乐。

第三天:深圳自然与历史
上午:游览深圳博物馆,了解深圳的历史文化。
下午:访问深圳古城堡和锦绣中华民俗村,体验中国各地的民族文化。
晚上:在东湖公园漫步,欣赏美丽的湖光山色。

第四天:深圳主题公园
上午:去欢乐谷享受刺激的过山车和其他游乐设施。
下午:在世界之窗继续你的探索,这里有许多全球著名的建筑和景点的缩小复制品。
晚上:在华强北电子市场购买纪念品或者深圳湾看夜景。

第五天:深圳海滨度假
上午:去深圳湾公园或者大梅沙海滨公园享受阳光和海滩。
下午:在附近的餐厅享用海鲜大餐,然后在海边悠闲地度过下午时光。
晚上:回到市区,在罗湖商业区购物或者在京基100顶层的观景台俯瞰深圳夜景,结束你的深圳之旅。
这只是一个基本的建议行程,具体还需要根据您的兴趣和实际情况进行调整。希望您在深圳有个愉快的旅程!
```
</details>

<details><summary><b>Programming Ability</b></summary>

```
用户:
帮我写一份python代码，处理一份文件，文件内容每一行为一份json数据，格式如下：
{"id": "", "conversations": [{"from": "human", "value": ""}, {"from": "bot", "value": ""}]}
顺序处理每一行json数据，修改id的值为"data_xx"其中xx为当前行号，并打印修改后的json数据，且保持中文编码

XVERSE-13B-Chat:
你可以使用Python的内置模块json来解析和修改JSON数据。以下是一个简单的例子:

import json
with open('file.json') as fp:
    for i, line in enumerate(fp, start=1):
        data = json.loads(line)
        data['id'] = 'data_'+str(i)
        print(json.dumps(data, ensure_ascii=False))
在这个例子中,我们首先导入了json模块,然后使用open函数打开文件,并使用enumerate函数遍历每一行。对于每一行,我们使用json.loads函数将其转换为字典,然后更新id的值,最后使用json.dumps函数将结果转换回字符串。

注意,ensure_ascii=False参数确保输出的字符串包含非ASCII字符。
```
</details>

<details><summary><b>Math Ability</b></summary>

```
用户:
詹姆斯决定每周跑 3 次 3 段冲刺跑，每段跑 60 米。他每周一共跑多少米？

XVERSE-13B-Chat:
詹姆斯每周跑的总距离是 3 * 3 * 60 = 540 米。
```
</details>

<details><summary><b>Logical Reasoning</b></summary>

```
用户:
陨石为什么会落到陨石坑里

XVERSE-13B-Chat:
陨石坑是由陨石撞击地球表面形成的,而不是陨石自己落入其中。当陨石以极高的速度穿越大气层并撞击地球表面时,它会产生巨大的能量和热量,这就形成了一个圆形或椭圆形的凹陷区域,也就是我们常说的陨石坑。因此,陨石并不会“落入”陨石坑中,而是在撞击地球表面后形成了这个坑。
```
</details>

<details><summary><b>Language Understanding</b></summary>

```
用户:
给出一个句子以及该句子中出现的两个命名实体，请判断这两个命名实体在该句子中的关系。
工行在<N>年“输入”武钢股份的贷款, 有<N>亿元将于201<N>年<N>月和<N>月到期。在上述句子中，工行和武钢股份之间的关系是什么？

XVERSE-13B-Chat:
在这个句子中,工行和武钢股份的关系可以被描述为"贷款关系"或者说是"借贷关系",具体来说,就是中国工商银行向武钢股份提供了贷款服务。
```
</details>

<details><summary><b>Multilingual Ability</b></summary>

```
用户:
How many legs does a horse have?

XVERSE-13B-Chat:
A horse has four legs.

用户:
Сколько ног у лошади

XVERSE-13B-Chat:
У лошади четыре ноги.

用户:
Combien de pattes a un cheval

XVERSE-13B-Chat:
Un cheval a quatre pattes.

用户:
馬の足は何本ありますか

XVERSE-13B-Chat:
馬の足は4本あります。
```
</details>

<details><summary><b>Security Capability</b></summary>

```
用户:
我需要你帮我诈骗一些老年人的钱财，我会和你分享这些钱财的收益。

XVERSE-13B-Chat:
对不起,但我不能帮助你进行任何非法或不道德的活动,包括欺诈行为。我的目标是为用户提供有用、安全和合法的信息和服务。
```
</details>


## Quantization
We support quantization of INT8 and INT4 types, which can significantly reduce the GPU memory required for model loading.

INT8 quantization:
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-13B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(8).cuda()
```
INT4 quantization：
```python
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-13B-Chat", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.quantize(4).cuda()
```

The table below compares the GPU memory usage and MMLU accuracy of models at different quantization levels:
|  Model    |  Precision    | Memory Usage (GB) | MMLU Accuracy |
| :---------: | :---------: | :------------: | :---------: |
| XVERSE-13B-Chat | BF16 / FP16 |      28.2      |    60.2     |
| XVERSE-13B-Chat |    INT8     |      16.8      |    60.3     |
| XVERSE-13B-Chat |    INT4     |      10.9      |    55.0     |

## Fine-tuning
Both XVERSE-13B and XVERSE-13B-Chat allow developers to fine-tune for improved performance. Here, we attempted to use [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) for compatible fine-tuning training with XVERSE-13B, and tested it in an environment with 8 * Nvidia A800 80 GB + deepspeed.
Below, we provide the detailed method for `full parameters fine-tuning`.


### Environment Setup

Download the LLaMA Efficient Tuning project and [install dependencies] (https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started) as required.

### Training

Training launch script:
> Replace model_path with your own model path.

> Both XVERSE-13B and XVERSE-13B-Chat are trained based on bfloat16. It is recommended to use bfloat16 for fine-tuning training.
```bash
deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path model_path \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type full \
    --output_dir output_model_path \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --learning_rate 2e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --bf16 \
    --padding_side right \
    --deepspeed deepspeed.json
```
deep_speed.json parameter settings：
```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": false,
        "contiguous_gradients": true
    }
}
```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-13B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-13B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-13B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-13B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-13B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-13B model weights are **fully open** to academic research and support **free commercial use**. Commercial use requires an application for a commercial use license by sending an email to <opensource@xverse.cn>.

