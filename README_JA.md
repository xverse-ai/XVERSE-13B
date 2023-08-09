<div align="center">
<h1>
  XVERSE-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-13B">🤗 Hugging Face</a>&nbsp ｜ &nbsp<a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README_EN.md">English</a> |
        <a href="README.md">中文</a> |
        <b>日本語</b>
    <p>
</h4>

## モデル紹介

**XVERSE-13B** は深圳遠翔科技が独自に開発した多言語大型言語モデルである。主な特徴は以下の通りです:

- **モデル構造**: XVERSE-13B は主流であるデコーダのみのトランスフォーマーネットワーク構造を採用し、同サイズのモデルの中で最長となる 8k のコンテキスト長をサポートしており、より長いマルチラウンド対話、知識質問応答、要約のニーズに応えることができる。これによって、このモデルはより汎用的な応用シナリオに対応できる。
- **トレーニングデータ**: このモデルは、中国語、英語、ロシア語、スペイン語など 40 以上の言語を含む、1.4兆個のトークンからなる多様で高品質なデータセットで徹底的に学習されています。異なる種類のデータのサンプリング比率が細かく設定されているため、中国語と英語の性能が優れており、他の言語の影響も考慮されている。
- **トークン化**: BPE（Byte-Pair Encoding）アルゴリズムに基づき、100,278 の語彙サイズを持つトークナイザーが、数百ギガバイトの言語データを用いて学習されました。このトークナイザは、追加の語彙拡張を必要とせず、多言語をサポートすることができます。
- **トレーニングフレームワーク**: 効率的な演算子、メモリの最適化、並列スケジューリング戦略、データ-計算-通信のオーバーラップ、プラットフォームとフレームワーク間の相乗効果など、いくつかの重要な技術も独自に開発されています。これらの進歩により、トレーニング効率とモデルの安定性が向上しました。これらの技術により、1,000 枚クラスタのピーク演算能力利用率は 58.5% に達し、業界の最先端を走っています。

## モデル評価

モデルの様々な能力を検証するために、[MMLU](https://arxiv.org/abs/2009.03300)(英語)、[C-Eval](https://cevalbenchmark.com/)(中国語)、[AGIEval](https://arxiv.org/abs/2304.06364)(中国語・英語)、[GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)(中国語・英語)、[GAOKAO-English](https://github.com/ExpressAI/AI-Gaokao)(英語)など、複数の分野にまたがる総合的な能力ベンチマークを選び、評価結果は以下の通りです:

|      モデル\データセット      |       MMLU       |      C-Eval      | AGIEval<sup>1</sup> | GAOKAO-Bench<sup>1</sup> | GAOKAO-English<sup>1</sup> |
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

> <sup>1: テストは単一解答の多肢選択問題のみで行われるため、穴埋め問題、自由形式問題、複数解答の多肢選択問題は除外される。</sup>
> <sup>2: [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) の結果を報告。</sup>
> <sup>3: [C-Eval](https://cevalbenchmark.com/) の結果を報告する。</sup>
> <sup>4: [Llama 2](https://arxiv.org/abs/2307.09288) の結果を報告。</sup>
>
> MMLU は、著者らが提供する[評価ツール](https://github.com/hendrycks/test)を採用し、C-Eval、AGIEval、GAOKAO-Bench、GAOKAO-English は MMLU と同じで、テストサンプルの構成は **5-shot** で統一する。

### MMLU カテゴリ結果
|        モデル\カテゴリ       | Average  |   STEM   | Social Science | Humanities |  Others  |
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

### C-Eval カテゴリ結果
|        モデル\カテゴリ       | Average  |   STEM   | Social Science | Humanities |  Others  |
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

### 環境設定

1. このリポジトリをクローンする:

```shell
git clone https://github.com/xverse-ai/XVERSE-13B
cd XVERSE-13B
```

2. pip を使って依存関係をインストールする:

```shell
pip install -r requirements.txt
```

### Transformers によるローディング

XVERSE-13B モデルは、以下のコードを用いて推論のためにロードすることができる:

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

### ウェブデモ

以下のコードはウェブサーバを起動するために使用することができます。ブラウザにアクセスアドレスを入力することにより、XVERSE-13B モデルによる推論を行うことができます:

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## 制限事項および免責事項

他の大規模言語モデル（LLM）と同様に、XVERSE-13B は特定の状況下で不正確、偏った、あるいは不快なコンテンツを生成する可能性があります。従って、モデルによって生成されたコンテンツを慎重に使用し、有害なコンテンツを広めないようにしてください。 XVERSE-13B のアプリケーションを展開する前に、開発者は安全性テストと特定のアプリケーションに応じたモデルの最適化を行う必要があります。

XVERSE-13B を利用して、有害な情報を作成・流布したり、公共性・国家性・社会性を損なったり、法規制に違反するような行為を行うことは、厳に慎んでください。XVERSE-13B モデルの使用により発生するいかなる問題（データセキュリティ上の問題、世論リスク、誤解、誤用、流布、コンプライアンス違反などによるリスクや問題）についても、当社は一切責任を負いません。

## オープンソースライセンス

このリポジトリにあるソースコードの使用は、[Apache-2.0](LICENSE) オープンソースライセンスに従う必要があり、XVERSE-13B のモデル重量の使用は、[モデルライセンス契約](MODEL_LICENSE.pdf)に従う必要があります。

XVERSE-13B のモデル分銅は、学術研究に対して**完全にオープン**であり、**自由な商用利用**をサポートしています。商用利用には、<opensource@xverse.cn> に電子メールを送って商用利用ライセンスを申請する必要があります。

