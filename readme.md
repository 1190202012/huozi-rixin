# 介绍

为任意的Frozen LLM进行检索增强，主要针对QA任务。

提出一种针对大模型的检索增强Pipeline框架。在输入上通过引入外源知识，提高大模型在事实性、时效性的表现，缓解缺乏领域知识的问题。在输出上使用拒绝采样方法，对单个query生成多个回复，通过多个回复之间的一致性或者有监督训练模型筛选回复，进一步提高模型回复的质量。

# 项目特点
本框架的特点包括：
1. 功能上：
  - 为任意大模型提供检索增强，即使是只提供API的大模型；
  - 除base模式外，还提供带有前端界面的demo模式、用于数据集评测的eval模型，并针对应用特点做了相应的并行优化；
2. 方法上：
  - 多样化来源的文档，包括：搜索引擎，维基百科，大模型生成。且支持增加其他来源的文档；
  - 两种知识处理方法：总结（summarize）和段落重排（contrive），提取文档中最相关的部分，使得其能被大模型更好的利用；
  - 不同的拒绝采样方法：针对不同的类型的数据集，我们提供了投票（voting）和打分（scoring）两种后处理的方法，对于答案简短、要求事实性的数据集，可以采用投票的方式对多个回复的一致性进行评分，选择一致性高的回复。对于答案较长、要求语言连贯、逻辑性的回复，我们分别训练了中、英文的奖励模型，对回复进行打分，选择分数最高的回复。
3. 实现上：
  - 在框架实现上做到了模块化，将检索、知识处理、拒绝采样的多种方法均以模块的方式提供调用接口，降低了模块间的耦合性；
  - 拓展性强，模块化的实现方式使得开发者可以方便的在检索、知识处理、拒绝采样部分拓展自己的方法；
  - 适合二次开发，开发者可以基于硬件情况，只需要更改配置文件，书写代码调用各模块，即可方便地实现个性化的需求。
  - 提供cache、log、ddp并行等功能，提高使用体验。

# Pipeline介绍
## 检索文档
对于输入query，首先根据来源的不同分别进行以下三种检索：
- web: 通过serper api调用google搜索引擎查询query，获取url list，然后根据url list获取对应网页的内容并进行简单的内容抽取；
- wiki：借助pyserini包检索相应的wiki条目；
- gen：使用大模型针对给定query生成文档。

## 知识构建
web、wiki两个来源的文档均有多个，可以选择多个文档拼接后进行以下处理：
- summarize: 使用大模型针对给定query生成拼接文档的摘要；
- contrive: 将文档进行分段后，对query和段落分别进行编码，然后计算相似度，根据相似度的高低选择部分段落。
大模型生成的文档一般不需要处理。

## 回复生成
基于得到的知识，使用大模型生成对应的回复。回复生成阶段采用拒绝采样方法，对于一个query，提供多个候选知识，分别生成多个候选回复。

## 拒绝采样
对于一个query，模型会得到多个回复，我们会采用以下方式从生成的多个回复选择合适的回复。
- vote：对于较短、要求事实性的回复，可以计算回复之间的一致性，决定是否回复并选择一致性高的回复。
- score: 对于较长、要求语言连贯、逻辑性的回复，可以调用我们训练的好的奖励模型对回复进行打分，选择分数最高的回复。

在没有回复长短、评价指标的情况下，我们目前默认对多个回复先进行vote筛选然后score得到最好的回复。。

# 数据集评测
我们选择了eli5（筛选436条query），truthfulqa（817条query），open natural question（3610条query）三个有代表性的数据集，其中eli5主要为生活常识类，答案不唯一，言之有理即可，注重模型的逻辑连贯性。truthfulqa同样为生活常识类，但大多具有唯一答案，主要评估模型的事实性。open natural question为事实性问题，提问多为“When”，“Who”，“Which”，“Where”，“How many”，属于知识密集型的QA任务，考察模型对知识的应用能力（open book， close book），知识的存储能力（close book）。

三个数据集的评测方式与结果如下所示。

## eli5 & truthfulqa
由于eli5与truthfulqa均是长回复，除人工评估外无更好的评估方式，我们会在后续更新该结果

## open natural question
+------------------------------------+------------+-------------+
| models                             |   em_score |   bem_score |
|------------------------------------+------------+-------------|
| contrive google no truncate merged |   0.300277 |    0.461997 |
| gen doc                            |   0.254294 |    0.408483 |
| no knowledge response              |   0.102216 |    0.243093 |
| summarize google good              |   0.322161 |    0.47798  |
| summarize google plus wiki         |   0.382548 |    0.537431 |
| system_answer                      |   0.360942 |    0.513192 |
+------------------------------------+------------+-------------+

我们在后续论文中进行了更多的实验，表明拒绝采样过程的有效性。活字-日新中同时使用了两种拒绝采样策略（voting与reward model）和仅四种 文档拼接和处理 方式，所以并未达到最好效果。目前方法组合未进行调优，后续会更新最新方法。

# 项目框架介绍

文件夹介绍如下：
- .vscode: vscode配置文件夹，在launch.json文件中保存有项目远程debug配置。
- aaai24: 项目代码文件
- config: 项目配置文件
- data: 数据
- log: log目录
- output: 输出目录，也是程序加载cache的目录

文件介绍如下：
- .gitignore
- LICENSE
- requirement.txt
- readme.md: 本文件
- Todo.md: 后续功能拓展
- openai_usage: openai接口使用情况
- run.py: 项目入口

重要文件和文件夹介绍

## run.py
项目入口。项目运行只需要两个参数：配置文件（必须），是否开启调试（可忽略）。

所有的配置文件均保存在config目录下（aaai24/config为解析config文件的代码）

在run.py中有三种模式: base, demo, eval。在config目录下yaml文件中设置。

base是在一个或多个样例上进行实验。

demo会调用gradio得到一个具有可视化界面的demo，采用torch.multiprocessing包提供多进程的支持。

eval模式会在QA数据上测试效果，目前支持eli5（从测试集中筛选部分高质量query），truthfulqa，open natural question三个数据集，采用torch.DistributeDataParallel包提供并行。demo和eval模型目前均为单机多卡。

## aaai24
该目录下保存有项目运行的全部代码。不在其中的代码比如处理数据集的一次性代码。

主要的运行代码：
- demo.py: 针对demo模式的代码。 
- eval.py: 针对数据集评估的代码。
- main.py: 基础代码，对应base模式。

模块化的包：
- config: 解析config文件，检查配置输入输出目录。
- data: 加载数据
- retrieval: 为query检索文档的代码，其中gen, web, wiki是三种检索方法的代码
- knowledge: 处理文档生成知识的代码，其中summarize, contrive是两种处理方法
- response: 生成回复的代码
- voting: 拒绝采样方法一，为生成的多个回复进行一致性投票
- scoring: 拒绝采样方法二，为生成的多个回复进行打分。
- utils: 提供大模型工具类LLM，所有的大模型初始化，生成，编码，打分操作均在该类中。包中提供调用openai api、PromptTemplate类、文本长度控制方法。

### retrieval
在base.py中提供BaseRetrieval抽象类，加载cache，保存输入等函数均在其中。

retriever初始化时，如果cache path不为空，为query进行检索时会首先在cache中进行查找，未找到时才会进行检索操作。

如果wiki相关的环境变量配置不成功或者索引文件过大（55G）难以下载，可以直接使用output目录中已有的cache文件。wiki_retriever采用Lazy initialization，只有当必须检索query时才会查找索引文件。

以下是三种retrieve方法在config文件中的配置，如果ban为True，那么该module（某个具体的方法初始化）便在查询时仅返回None，不读入cache或检查cache文件是否存在。load_result_file和load_info_file是cache文件名，项目会加载cache_dir/load_{result,info}_file作为该module的cache。save_result_file和save_info_file是该module保存结果的文件名，保存在output_dir/{time}/{rank if ddp}/save_{result|info}_file下。

```yaml
Web:
    ban: False
    load_result_file: web_doc.json
    save_result_file: web_doc.json
    load_info_file: web_info.json
    save_info_file: web_info.json
    log_detail: False
    ssl_verify: False # 减少获取网页文档时的报错
    min_doc_len: 50 # 文档最短长度，过滤后的文档低于该长度会返回None
    max_doc_len: 1000 # 文档最长长度，高于该长度会截断并去掉不完整的句子


Wiki:
    ban: False
    load_result_file: wiki_doc.json
    save_result_file: wiki_doc.json
    load_info_file: wiki_info.json
    save_info_file: wiki_info.json
    log_detail: False


Gendoc:
    ban: False
    load_result_file: gen_doc.json
    save_result_file: gen_doc.json
    load_info_file: gendoc_info.json
    save_info_file: gendoc_info.json
    log_detail: False
    zh_model_name: chatglm2 # 中文生成文档选用的模型
    zh_template_id: 1 # 中文生成文档选用的prompt
    en_model_name: llama2-chat # 英文生成文档选用的模型
    en_template_id: 1 # 英文生成文档选用的prompt
    tokenize_kwargs: # 传入tokenizer的参数
        padding: "longest" 
        truncation: True
    generate_kwargs: # 传入model.generate的参数，batch_size会pop out。
        batch_size: 24
        temperature: 0.7
        top_p: 0.8
        top_k: 50
        repetition_penalty: 1.3
        do_sample: False

```

### knowledge
在base.py中提供BaseConstructor类，其作用类似于BaseRetriever。

contriver和summarize包中分别提供contrive和summarize两种处理文档的方法。在contrive中对于英文我们使用webglm的query_encoder和paragraph_encoder分别对query和段落进行编码，对于中文我们使用[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)对query和段落进行编码。

以下是两种方法config文件。

```yaml
Summarizer:
    ban: False
    load_result_file: # 为空表示不读取cache。cache_dir为空同样可以达到效果，其会不加载所有的cache
    save_result_file: summarize_result.json
    load_info_file: 
    save_info_file: summarize_info.json
    zh_model_name: chatglm2
    zh_template_id: 3
    en_model_name: llama2-chat
    en_template_id: 4
    tokenize_kwargs:
      padding: "longest"
      truncation: True
    generate_kwargs:
      batch_size: 24
      temperature: 0.8
      top_p: 0.9
      top_k: 50
      repetition_penalty: 1.2
      do_sample: False


Contriver:
    ban: False
    load_result_file: contrive_result.json
    save_result_file: contrive_result.json
    load_info_file: contrive_info.json
    save_info_file: contrive_info.json
    min_knowledge_len: 300
    tokenize_kwargs:
        padding: "longest"
        truncation: True
        max_length: 512
    encode_kwargs: # 编码时传入的参数，batch_size和pooling_method均会pop out
        batch_size: 24 
        pooling_method: mean # 池化方法，候选方法有mean, max, sum, cls

```

### response
其配置文件如下：

```yaml
Generator:
    ban: False
    load_result_file: 
    save_result_file: response_result.json
    load_info_file: 
    save_info_file: response_info.json
    zh_model_name: xverse-chat
    zh_template_id: 3
    en_model_name: llama2-chat
    en_template_id: 9
    tokenize_kwargs:
        padding: "longest"
        truncation: True
    generate_kwargs:
        batch_size: 16
        temperature: 0.8
        top_p: 0.9
        top_k: 50
        repetition_penalty: 1.2
        do_sample: False
        max_new_tokens: 300 # 最大生成长度，大模型会在生成长度超过300时强制截断，即使没有生成终止符号。并且会依据此长度和模型最大输入长度对输入+知识进行截断
```

### voting
采用bert_score的计算多个回复的一致性，配置文件如下：
```yaml
Voter:
    ban: False
    load_result_file: 
    save_result_file: vote_result.json
    load_info_file: 
    save_info_file: vote_info.json
    tokenize_kwargs: 
      padding: "longest"
      truncation: False
    encode_kwargs:
      batch_size: 24
```

### scorer
奖励模型使用[deepspeed-chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)训练而成，其中英文模型采用llama2-chat在[openai-webgpt_comparasion](https://huggingface.co/datasets/openai/webgpt_comparisons)中eli5数据集部分训练而成，中文模型采用XVRESE-13B在[beyond/rlhf-reward-single-round-trans_chinese](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)数据集上训练得到。

配置文件如下:

```yaml
Scorer:
    ban: False
    load_result_file: 
    save_result_file: score_result.json
    load_info_file: 
    save_info_file: score_info.json
    zh_model_name: xverse_base_reward_model
    zh_template_id: 1
    en_model_name: llama2_base_reward_model
    en_template_id: 1
    tokenize_kwargs: 
      padding: "longest"
      truncation: False
    reward_kwargs:
      batch_size: 16
```

### utils
- llm.py: 提供LLM类，大模型初始化后以{name: {model,tokenizer}}的形式存入LLM.llms中，该类提供初始化大模型、释放大模型、生成、编码、计算奖励值函数，生成、编码、奖励值方法均需要传入gpu；
- prompt_template.py: PromptTemplate类，传入带有placeholder的字符串后可以调用该类进行填充；
- openai_tools.py: 调用openai api的方法，在使用前你需要设置自己的API KEY，使用量会记录在openai_usage.jsonl中；
- modeling_reward_model.py: 奖励模型结构；
- doc.py: 将文档剪切到给定长度。

## config
包含所有config文件，module配置部分介绍见上文。

首先以demo.yaml为例来看大模型配置部分
```yaml
LLMConfig:
  llm1:
    model_name: llama2_base_reward_model # 模型名称，通过LLM.llm_config_dict[name]可以获取对应的配置
    model_path: ./data/model/llama2_base_reward_model/pytorch_model.bin # 模型路径，可以是本地或者huggingface。不提供"tokenizer_path"时以model_path作为替代
    model_class: LlamaModel # model class
    fp16: True # 是否开启fp16
    tokenizer_class: LlamaTokenizer # tokenizer class
  llm2:
    model_name: llama2-chat
    model_path: daryl149/llama-2-7b-chat-hf
    model_class: LlamaForCausalLM
    fp16: True
    tokenizer_class: LlamaTokenizer
  llm3:
    model_name: xverse-chat
    model_path: ./data/model/xverse_chat
    model_class: LlamaForCausalLM
    fp16: True
    tokenizer_class: AutoTokenizer
  llm4:
    model_name: xverse_base_reward_model
    model_path: ./data/model/xverse_base_reward_model/pytorch_model.bin
    model_class: LlamaModel
    fp16: True
    tokenizer_class: AutoTokenizer
  llm5:
    model_name: en_query_encoder
    model_path: ./data/model/webglm_dual_encoder/query_encoder
    model_class: AutoModel
    fp16: False
    tokenizer_path: facebook/contriever-msmarco
    tokenizer_class: AutoTokenizer
  llm6:
    model_name: en_paragraph_encoder
    model_path: ./data/model/webglm_dual_encoder/paragraph_encoder
    model_class: AutoModel
    fp16: False
    tokenizer_path: facebook/contriever-msmarco
    tokenizer_class: AutoTokenizer
  llm7:
    model_name: chatglm2
    model_path: THUDM/chatglm2-6b
    model_class: AutoModel
    fp16: True
    tokenizer_class: AutoTokenizer
  llm8:
    model_name: bert-base-chinese
    model_path: bert-base-chinese
    model_class: BertModel
    fp16: False
    tokenizer_class: AutoTokenizer
  llm9:
    model_name: deberta
    model_path: microsoft/deberta-xlarge-mnli
    model_class: AutoModel
    fp16: False
    tokenizer_class: AutoTokenizer
  llm10:
    model_name: zh_contrive_encoder
    model_path: BAAI/bge-large-zh
    model_class: AutoModel
    fp16: False
    tokenizer_class: AutoTokenizer
  llm11:
    model_name: huozi-rlhf
    model_path: HIT-SCIR/huozi-7b-rlhf
    model_class: AutoModelForCausalLM
    fp16: True
    tokenizer_class: AutoTokenizer
    padding_side: left
```

下面来看输入输出配置部分。分别来看eval.yaml、base.yaml、demo.yaml的输入输出文件配置。

其中eval.yaml针对eval模式，eval模式下为了提高评估速度建议开启多进程，即设置ddp=True。此时会有多个进程将输入分别写入多个子文件夹，最后由进程0合并在一起。

demo.yaml默认不提供输入输出路径，但是模型仍会在内存中保存信息和处理过的结果，所以如果demo长时间运行可能因此OOM。

base.yaml并不使用并行。

```yaml
# base.yaml
data: truthfulqa # yaml文件中的占位符，代码中并未使用该属性，只会将当前yaml文件中{data}进行填充
test_file: ./data/datasets/{data}/{data}_cleaned.jsonl # 数据路径。Todo: 为了避免困惑，这里最好设置为./data/datasets/examples/truthfulqa.jsonl, 但写这个文档时当时的cache_dir丢失了

cache_dir: ./output/experiment/2023-08-21-16-42-15/ # 从该路径下寻找保存的文件

# 若cache_dir设为空，即使后续load_{file|result}_file不为空，也不会读取cache
# cache_dir: 

output_dir: ./output/experiment/ # output_dir在读取config文件时会在后续加上当前时间。即真正的output_dir为output_dir + "%Y_%m_%d_%H_%M_%S/"，而output_dir和save_{result|info}_file拼接后即为module保存中间信息和结果的路径

result_file: result.json # 最终结果也会被保存在output_dir + "%Y_%m_%d_%H_%M_%S/" + result_file中
result_info_file: result_info.json

# gpu id
gpu: 0

ddp: False # base模式建议ddp设为False.
```

```yaml
# demo.yaml
data:
test_file: # demo仅从前端得到query 

cache_dir: # demo启动时一般不设置cache_dir

output_dir: # demo一般不保存用户查询信息

result_file: # 一般不设置
result_info_file: # 一般不设置

# 80g gpu id
gpu: 0, 1, 2, 3 # demo.yaml适合4张80g显存的卡

ddp: False # demo模式使用torch.multiprocessing包手动进行并行，不使用ddp
```

eval.yaml此时开启了ddp，代码中会进行额外的处理

```yaml
# eval.yaml
data: truthfulqa # 使用占位符，方便更换数据集
test_file: ./data/datasets/{data}/{data}.jsonl

cache_dir: ./output/eval/{data}_long/ # 可以以之作为cache目录，或者其他目录

output_dir: ./output/eval/{data}/ # 由于设置ddp=True，输出会保存至output/eval/data/%Y_%m_%d_%H_%M_%S/{rank}/，其中rank是进程序号，范围是[0,gpu_num - 1]，最后会由进程0合并所有进程的输出保存在output/eval/data/%Y_%m_%d_%H_%M_%S/

result_file: result.json # 类似于module_config["save_result_path"]，也会先由各进程保存至output/eval/data/%Y_%m_%d_%H_%M_%S/{rank}/{result_file}, 最后进行合并
result_info_file: result_info.json

# gpu id list. If set None, will use gpu in [0, torch.cuda.device_count() - 1]
gpu: 

# torch.DistributeDataParallel
ddp: True
```

## data

数据集可以放在data/examples，data/dataset下，只需要在config下yaml文件设置好即可。

但如果要在eval模式下使用，数据集中必须包含三个列，“query”, "language", "truthful answer"，demo模式不需要数据集，base模式至少需要“query”, "language"。

## output

output目录下保存有运行文件，建议将save_{result|info}_file设置为非空，由于time作为子目录名，一般不会覆盖掉之前存在的文件夹，只需要注意服务器的时间可能与本地时间不一致。这样做的好处是通过设置cache_dir和load_{result|info}_path模型可以很多的完成之前已经完成的部分。


# 环境配置
## 整体流程
以下流程在ubuntu 22.04中执行，架构为x64。如果你使用其他操作系统或者架构，请注意甄别。
1. 安装Conda环境，可以使用[miniconda](https://docs.conda.io/projects/miniconda/en/latest/)。
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```
    退出当前命令行，重新进入

2. 新建python环境。
    ```bash
    conda create -n rixin python=3.9
    ```
3. 安装pytorch。
    ```bash
    pip3 install torch
    ```
4. 安装JDK。
    ```bash
    apt install default-jdk
    ```
    如果报错可以输入以下命令。
    ```bash
    apt-get update
    apt install default-jdk --fix-missing
    ```
    继续以下步骤（无论上面是否报错都需要这一步）
    ```bash
    conda install -c conda-forge openjdk=11
    ```

5. 安装mkl包，用于解决[来自Faiss 1.7.4的一个bug](https://github.com/facebookresearch/faiss/issues/2890)，后续可能修复。
    ```bash
    conda install mkl=2021
    ```

6. 安装requirement.txt中的包。
    ```bash
    cd YOUR_PROJECT_DIR
    pip install -r ./requirement.txt
    ```

7. 安装faiss包：
    ```
    conda install -c pytorch faiss-cpu=1.7.4
    ```

## 模型文件和数据文件
1. 需要从Huggingface下载的大模型文件：
  - daryl149/llama-2-7b-chat-hf
  - xverse/XVERSE-13B-Chat
  - THUDM/chatglm2-6b
  - bert-base-chinese
  - microsoft/deberta-xlarge-mnli
  - BAAI/bge-large-zh
  - facebook/dpr-question_encoder-multiset-base
  - HIT-SCIR/huozi-7b-rlhf

2. 需要从Huggingface下载的数据文件：
  - eli5
  - truthful_qa
  - nq_open

3. 其他需要下载的模型和文件（在docker、hpc（HIT-SCIR内部服务器）、cfs（XVERSE内部服务器）中项目目录下均有保存，以下仅是获取方式）：
  - Wiki英文索引文件。首先配置好pyserini包，教程在下文。进入python命令行后输入以下代码进行下载。（大约50G，从加拿大滑铁卢大学服务器下载，注意代理，索引文件保存在~/.cache/pyserini/indexes）
    ```python
    from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
    en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))
    ```
  - Wiki中文索引文件。参照[FlagEmbedding Search Demo样例](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/search_demo/readme.md)，只需要完成Data Preprocessing and Indexing中1，3部分。需要对其代码进行以下修改：
    1. arguments.py中"BAAI/bge-large-zh-noinstruct" -> "BAAI/bge-large-zh"
    2. 注释preprocss.py的104行，BM25粗排只会召回政治敏感信息；  
  运行完preprocss.py后data目录下collection/documents.json为wiki文本，emb目录下为编码后的两百万条wiki条目的表示。复制这两个文件至本项目目录下data/index即可

  - webglm_dual_encoder. 来自[WebGLM](https://github.com/THUDM/WebGLM)，[query_encoder和paragraph_encoder下载链接](https://cloud.tsinghua.edu.cn/d/bc96946dd9a14c84b8d4/)

  - 奖励模型，中文奖励模型基于XVERSE-7B在beyond/rlhf-reward-single-round-trans_chinese数据集上使用DeepSpeed-Chat框架训练而成，英文奖励模型使用daryl149/llama-2-7b-chat-hf基于openai/webgpt_comparisons中**eli5部分且answer_0和answer_1不相等**部分使用DeepSpeed-Chat框架炼制而成。[原训练代码链接](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning)或使用本项目中的复制版（train_reward_model目录下）


## 对于部分包的说明
1. gradio如果要提供公网地址（即直接采用demo.launch(share=True)获取公网可访问地址），经过测试需要挂代理。

2. 搜索引擎检索调用Web API目前使用serper api, 需要自行替换API KEY，目前官网注册可以有2,500次免费次数。同样可以直接采用搜索引擎URL拼接query的方式进行。

3. 获取URL之后需要根据URL获取页面内容，目前使用异步的方式提高爬取速度

4. Wiki英文检索中使用pyserini包，[安装教程](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
  1. 安装JDK11：conda install -c conda-forge openjdk=11
  2. 安装torch或者确认torch安装
  3. pip install pyserini
  4. conda install -c pytorch faiss-cpu=1.7.4 mkl=2021
  5. 执行上述命令下载索引文件
  6. 测试安装是否成功
  ```python
  from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
  import json
  en_searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-multi-bf', DprQueryEncoder('facebook/dpr-question_encoder-multiset-base'))
  query = "Who is Elon Mask"
  hits = self.en_searcher.search(query)
  doc = json.loads(self.en_searcher.doc(hits[i].docid).raw())["contents"]
  print(doc)
  ```


# 运行

## base
单卡即可。注意显存和内存是否满足需求。

``` bash
python run.py -c config/demo.yaml
```

可以在命令行前加上HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1表示使用本地已保存的dataset和model


## demo
必须有2张40G内存的卡，否则请改写aaai24/demo.py和config/demo.yaml
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py -c config/demo.yaml
```

前面的“HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1”仅是建议，非必须。

## eval
根据卡的现状，改写aaai24/eval.py和config/eval.yaml

默认使用全部的卡。否则请在config/eval.yaml中指定要使用的gpu

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py -c config/eval.yaml
```

前面的“HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1”仅是建议，非必须。


# Debug
本项目代码可以简单方便地在debug模式下运行，方便熟悉项目代码以及开发过程中的Debug。主要配置文件分别为.vscode/launch.json和run.py中line 18-21。

如果不需要debug，只需要注意命令行参数中不要有"-d"或则"--debug"。

主要适用场景为：项目必须在某个GPU结点上运行，但是无法ssh到该结点或者使用的GPU结点并不固定，不方便使用vscode remote打开GPU结点的文件目录。但是你有一个固定的跳板机，跳板机与GPU的项目文件是自动同步的或者就是同一块硬盘存储区域，并且两台机器在一个局域网下。此时你可以用vscode+remote插件打开在跳板机上的项目目录。然后设置.vscode/launch.json。

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0", // 无须改动
    "configurations": [
        {
            "name": "Python: 远程调试", // 无须改动
            "type": "python", // 无须改动，但是注意你的vscode界面右下角一般会显示你当前使用的Python环境，以便可以对编辑界面进行解析。如果右下角只显示文件类型为python，没有显示Python环境，并且平时编辑代码vscode并没有提示，说明vscode并没有正确地识别到python环境，这会导致调试失败，提示不支持当前type，即使此时项目可以在命令行运行。
            "request": "attach", // 无需改动，这一配置表示你的调试是依托于远程运行的程序的。
            "listen": {
                "host": "0.0.0.0", // 无需改动，表示启动调试后会跳板机会监听本机的某个端口，该端口会收GPU结点发送的调试信息，从而你可以在跳板机上看到程序运行情况。
                "port": 6789 // 端口号，注意不要与已有端口冲突，与run.py中debugpy.connect的端口保持一致
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}", //远程调试并不需要在跳板机指定某个文件，而是GPU结点运行到某个文件，vscode就打开跳板机上对应的某个文件。所以只需要指定远程GPU结点的目录与作为本地的跳板机上的哪个目录是对应的。
                    "remoteRoot": "."
                }
            ],
            "justMyCode": false, //是否进入import的包
        }
    ]
}
```

除了配置.vscode/launch.json，还需要run.py中以下语句：

```
debugpy.connect(('192.168.1.50', 6789)) # 与跳板机链接，"192.168.1.50"是跳板机内网IP，6789是跳板机接收调试信息的端口
debugpy.wait_for_client() # 等待跳板机的相应
debugpy.breakpoint() # 断点。一般而言直接在vscode界面上打断点即可。
```

配置好以上配置文件后，在已经打开了跳板机上项目目录的vscode上，选项栏“运行”——>“调试”，此时vscode保持监听状态，然后在gpu结点上命令行运行程序，顺利的话vscode会打开run.py文件停留在line23等待你继续调试。

开启debug模式只需要在命令行中加入'-d'或'--debug'参数即可，开启debug模式需要跳板机打开调试监听对应端口，否则GPU结点上命令行运行的程序也会因为没有收到跳板机的响应而报错。