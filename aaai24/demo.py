from uuid import uuid4
from aaai24.retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from aaai24.knowledge import SummarizeConstructor, ContriverConstructor
from aaai24.response import StandardGenerator
from aaai24.voting import StandardVoter
from aaai24.scorer import RewardScorer
from aaai24.utils import LLM, truncate_en_doc, truncate_zh_doc
from torch.multiprocessing import Process
from torch.multiprocessing import Manager
import gradio as gr
import traceback

# for gradio web ui
CSS = """
    #left-col {
        width: 30%;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        margin: auto;
    }
    
    #right-col {
        min-width: 45%;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        margin: auto;
    }
    
    footer{display:none !important}
"""

doc_html = """

<details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
    <summary style="display: flex; align-items: center; font-weight: bold;">
        <span style="margin-right: 10px;">[{index}] {title}</span>
        <a href="{url}" style="text-decoration: none; background: none !important;" target="_blank">
            <!--[Here should be a link icon]-->
            <i style="border: solid #000; border-width: 0 2px 2px 0; display: inline-block; padding: 3px; transform:rotate(-45deg); -webkit-transform(-45deg)"></i>   
        </a>
    </summary>
    <p style="margin-top: 10px;">{text}</p>
</details>

"""

model_name2url = {
    "llama2-chat": "https://ai.meta.com/llama/",
    "xverse-chat": "https://github.com/xverse-ai/XVERSE-13B",
    "chatglm2": "https://github.com/THUDM/ChatGLM2-6B",
}


# for multiprocessing
def _update(task_map, gpu, uuid, task_item):
    temp = task_map[gpu]
    temp.update({uuid: task_item})
    task_map[gpu] = temp


def _remove(task_map, gpu, uuid):
    temp = task_map[gpu]
    ret = temp.pop(uuid)
    task_map[gpu] = temp
    return ret


def run(rank, config, task_map, wiki_retriever, gen_retriever, summarizer, contriver, generator, voter, scorer):
    try:
        gpu = f"gpu{rank}"

        LLM.get_llm_config(config["LLMConfig"])
        LLM.gpu_ids = config["gpu"]
        LLM.ddp = config["ddp"]

        LLM.initial_all(gpu, config["LLMMap"][gpu])

        if rank == 0:
            wiki_retriever.initialize()

        # gpu: {uuid: (task, input, output)}
        task_map[gpu] = {}

        while True:
            uuid_list = list(task_map[gpu].keys())

            for uuid in uuid_list:
                task_item = task_map[gpu].get(uuid, None)

                if task_item is not None and task_item[2] is None:
                    task, kwargs, _ = task_item
                    if task == "wiki":
                        result = wiki_retriever.retrieve(**kwargs)
                    elif task == "gendoc":
                        result = gen_retriever.retrieve(**kwargs)
                    elif task == "summarize":
                        result = summarizer.batch_construct(**kwargs)
                    elif task == "contrive":
                        result = contriver.construct(**kwargs)
                    elif task == "response":
                        result = generator.batch_response(**kwargs)
                    elif task == "vote":
                        result = voter.voting(**kwargs)
                    elif task == "score":
                        result = scorer.batch_score(**kwargs)
                    else:
                        assert False
                    _update(task_map, gpu, uuid, (task, kwargs, result))
    except:
        print(f"{rank}报错")
        print(traceback.format_exc())
        exit(-1)


def build_demo(config, debug=False):
    # initial module and language model
    task_map = Manager().dict()

    web_retriever = WebRetriever(config["ModuleConfig"]["Web"])
    wiki_retriever = WikiRetriever(config["ModuleConfig"]["Wiki"])
    gen_retriever = GenerationRetriever(config["ModuleConfig"]["Gendoc"])

    summarizer = SummarizeConstructor(config["ModuleConfig"]["Summarizer"])
    contriver = ContriverConstructor(config["ModuleConfig"]["Contriver"])

    generator = StandardGenerator(config["ModuleConfig"]["Generator"])
    voter = StandardVoter(config["ModuleConfig"]["Voter"])
    scorer = RewardScorer(config["ModuleConfig"]["Scorer"])

    process_list = []

    for i in range(len(config["LLMMap"])):
        process = Process(target=run, args=(
        i, config, task_map, wiki_retriever, gen_retriever, summarizer, contriver, generator, voter, scorer))
        process_list.append(process)
        process.start()

    while (len(task_map) < len(config["LLMMap"])):
        continue

    # demo executor
    def demo_query(query, language):
        try:
            _dict = {}
            # retrieval
            kwargs = {"query": query, "language": language, "gpu": "gpu1"}
            task_item = ("gendoc", kwargs, None)
            uuid_1 = str(uuid4())
            _update(task_map, "gpu1", uuid_1, task_item)

            kwargs = {"query": query, "language": language, "gpu": "gpu0"}
            task_item = ("wiki", kwargs, None)
            uuid_2 = str(uuid4())
            _update(task_map, "gpu0", uuid_2, task_item)

            web_docs = web_retriever.retrieve(query, language)

            kwargs = {"query": query, "language": language, "doc": "\n".join(web_docs), "gpu": "gpu0", "top_k": 10}
            task_item = ("contrive", kwargs, None)
            uuid_3 = str(uuid4())
            _update(task_map, "gpu0", uuid_3, task_item)

            while task_map["gpu1"][uuid_1][2] is None or task_map["gpu0"][uuid_2][2] is None or task_map["gpu0"][uuid_3][2] is None:
                continue

            gen_doc = _remove(task_map, "gpu1", uuid_1)[2]
            wiki_docs = _remove(task_map, "gpu0", uuid_2)[2]
            contrive_google_10 = _remove(task_map, "gpu0", uuid_3)[2]

            #all_sources
            all_sources = "\n".join(web_docs) + "\n" + "\n".join(wiki_docs) + "\n" + gen_doc

            _dict["docs"] = []
            info_dict = web_retriever.info_dict[query]
            url_item_list, url2doc_dict = info_dict["search"]["organic"], info_dict["fetch"]
            url2item_dict = {item["link"]: item for item in url_item_list}
            doc2url_dict = {doc: url for url, doc in url2doc_dict.items()}
            for doc in web_docs:
                url = doc2url_dict[doc]
                item = url2item_dict[url]
                _dict["docs"].append({"title": item["title"], "url": url, "text": doc})

            for doc in wiki_docs[:2]:
                _dict["docs"].append({"title": "Wikipedia Item",
                                      "url": "https://en.wikipedia.org/wiki/Main_Page" if language == "en" else
                                      "https://zh.wikipedia.org/wiki/Wikipedia:%E9%A6%96%E9%A1%B5",
                                      "text": doc})

            _dict["docs"].append({"title": "LLM Generate Doc",
                                  "url": model_name2url[config["ModuleConfig"]["Gendoc"][f"{language}_model_name"]],
                                  "text": gen_doc})
            yield _dict

            # knowledge
            # select_num = min(len(web_docs), 3)
            # truncate_doc = truncate_zh_doc if language == "zh" else truncate_en_doc
            # _temp_doc_0 = "\n".join([truncate_doc(web_docs[i], 1200 // select_num) for i in range(select_num)]
            #                       + [truncate_doc(wiki_docs[i], 150) for i in range(2)])
            # _temp_doc_1 = web_docs[0] if len(web_docs) > 0 else ""
            temp_doc_0 = contrive_google_10
            select_num = min(len(web_docs), 2)
            truncate_doc = truncate_zh_doc if language == "zh" else truncate_en_doc
            #gm_w
            google_merge_plus_wiki = "\n".join([truncate_doc(web_docs[i], 250) for i in range(select_num)]
                                  + [wiki_docs[i] for i in range(5)])


            kwargs = {"queries": query, "language_list": [language], "doc_list": [temp_doc_0], "gpu": "gpu1"}
            task_item = ("summarize", kwargs, None)
            uuid_1 = str(uuid4())
            _update(task_map, "gpu1", uuid_1, task_item)

            kwargs = {"query": query, "language": language, "doc": all_sources, "gpu": "gpu0", "top_k": 10}
            task_item = ("contrive", kwargs, None)
            uuid_2 = str(uuid4())
            _update(task_map, "gpu0", uuid_2, task_item)

            kwargs = {"query": query, "language": language, "doc": "\n".join(web_docs), "gpu": "gpu0", "top_k": 5}
            task_item = ("contrive", kwargs, None)
            uuid_3 = str(uuid4())
            _update(task_map, "gpu0", uuid_3, task_item)

            while task_map["gpu1"][uuid_1][2] is None or task_map["gpu0"][uuid_2][2] is None or task_map["gpu0"][uuid_3][2] is None:
                continue

            #gc_sum
            summarize_google_contrive_list = _remove(task_map, "gpu1", uuid_1)[2]
            summarize_google_contrive = "\n".join(summarize_google_contrive_list)
            #c_all
            contrieve_all_sources = _remove(task_map, "gpu0", uuid_2)[2]
            #gc_w !!!!!!!!!!!!!!!!!!!!!!
            contrive_google_5 = _remove(task_map, "gpu0", uuid_3)[2]
            google_contrieve_plus_wiki = "\n".join([contrive_google_5] + wiki_docs[:5])
            #wiki
            wiki = "\n".join(wiki_docs[:10])
            

            _dict["knowledge"] = [google_contrieve_plus_wiki, google_merge_plus_wiki, wiki, summarize_google_contrive, contrieve_all_sources]
            yield _dict

            # response
            from datetime import datetime
            
            if language == "en":
                time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")
                time_prompt = "The current time is {time}.\n"
            else:
                time = datetime.now().strftime("%Y年%m月%d日 %A %H点%M分")
                time_prompt = "当前时间是{time}。\n"
                weekday = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}

                for k,v in weekday.items():
                    time_prompt = time_prompt.replace(k,v)
            
            doc_list = [google_contrieve_plus_wiki, google_merge_plus_wiki, wiki, summarize_google_contrive, contrieve_all_sources]

            for i in range(len(doc_list)):
                doc_list[i] = time + doc_list[i]
   
            uuid_1 = str(uuid4())
            kwargs = {"queries": [query] * 3, "language_list": [language] * 3, "knowledge_list": doc_list[:3], "query_only": False,
                      "gpu": "gpu0"}
            task_item = ("response", kwargs, None)
            _update(task_map, "gpu0", uuid_1, task_item)

            uuid_2 = str(uuid4())
            kwargs = {"queries": [query] * 2, "language_list": [language] * 2, "knowledge_list": doc_list[3:], "query_only": False,
                      "gpu": "gpu1"}
            task_item = ("response", kwargs, None)
            _update(task_map, "gpu1", uuid_2, task_item)

            uuid_3 = str(uuid4())
            kwargs = {"queries": [query], "language_list": [language], "knowledge_list": [' '], "query_only": True, 
                      "gpu": "gpu1"}
            task_item = ("response", kwargs, None)
            _update(task_map, "gpu1", uuid_3, task_item)
            
            while task_map["gpu0"][uuid_1][2] is None or task_map["gpu1"][uuid_2][2] is None or task_map["gpu1"][uuid_3][2] is None:
                continue

            response_list = _remove(task_map, "gpu0", uuid_1)[2] + _remove(task_map, "gpu1", uuid_2)[2] + _remove(task_map, "gpu1", uuid_3)[2]

            for i in range(len(response_list)):
                if len(response_list[i].replace(" ", "")) == 0 or response_list[i] == query:
                    response_list[i] = "Sorry, I don't know about this." if language == "en" else "很抱歉，我不知道。"

            _dict["responses"] = response_list

            yield _dict

            # voting
            kwargs = {"query": query, "language": language, "responses": response_list, "gpu": "gpu1"}
            task_item = ("vote", kwargs, None)
            uuid = str(uuid4())
            _update(task_map, "gpu1", uuid, task_item)

            while task_map["gpu1"][uuid][2] is None:
                continue

            vote_score_list = _remove(task_map, "gpu1", uuid)[2]

            threshold = 0.85

            chosen_status_list = ["Candidate" if vote_score_list[i] > threshold else "Reject" for i in range(6)]
            select_response_list = [response_list[i] for i in range(6) if vote_score_list[i] > threshold]
            select_response_indexes = [i for i in range(6) if vote_score_list[i] > threshold]

            answer = None
            if len(select_response_list) == 0:
                chosen_status_list = ["Candidate" for _ in range(6)]
                select_response_list = response_list
                select_response_indexes = [i for i in range(6)]
            elif len(select_response_list) == 1:
                chosen_status_list[chosen_status_list.index("Candidate")] = "Answer"
                answer = select_response_list[0]

            response_list_with_status = [chosen_status_list[i] + "\n" + response_list[i] for i in range(6)]
            _dict["responses"] = response_list_with_status
            yield _dict

            if answer is not None:
                _dict["responses"] = response_list_with_status
                _dict["answer"] = answer
                yield _dict
                return

            # score
            # gpu = "gpu0" if language == "zh" else "gpu1"
            gpu = "gpu0"
            kwargs = {"queries": [query] * len(select_response_list),
                      "language_list": [language] * len(select_response_list), "response_list": select_response_list,
                      "gpu": gpu}
            task_item = ("score", kwargs, None)
            uuid = str(uuid4())
            _update(task_map, gpu, uuid, task_item)

            while task_map[gpu][uuid][2] is None:
                continue

            score_list = _remove(task_map, gpu, uuid)[2]

            answer = select_response_list[score_list.index(max(score_list))]
            chosen_status_list[select_response_indexes[score_list.index(max(score_list))]] = "Answer"

            response_list_with_status = [chosen_status_list[i] + "\n" + response_list[i] for i in range(6)]
            _dict["responses"] = response_list_with_status
            _dict["answer"] = answer
            yield _dict
        except:
            yield "Error, Please try another query", "Error, Please try another query", "Error, Please try another query", "Error, Please try another query", "Error, Please try another query"

    # wrapper between front end and back end
    def _wrapper(query, progress=gr.Progress()):
        docs = []
        knowledge = ["Loading ...", "Loading ...", "Loading ...", "Loading ...", "Loading ..."]
        responses = ["Loading ...", "Loading ...", "Loading ...", "Loading ...", "Loading ...", "Loading ..."]
        answer = "Loading ..."

        yield [answer] + ["<h2> Fetching Docs </h2>"] + knowledge + responses

        language = "en"
        for _char in query:
            if '\u4e00' <= _char <= '\u9fa5':
                language = "zh"
                break

        for resp in demo_query(query, language):
            if "docs" in resp:
                docs = resp["docs"]
            if "answer" in resp:
                answer = resp["answer"]
            if "responses" in resp:
                responses = resp["responses"]
            if "knowledge" in resp:
                knowledge = resp["knowledge"]

            yield [answer] + ["<h2> Docs (Click to Expand)</h2>" + "\n".join(
                [doc_html.format(**item, index=idx + 1) for idx, item in enumerate(docs)])] + knowledge + responses

    # demo web framework based on gradio 
    with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:
        with gr.Row():
            with gr.Column(elem_id='left-col'):
                gr.Markdown(
                    """
                # 活字-日新
                """)
                with gr.Row():
                    query_box = gr.Textbox(show_label=False, placeholder="Enter question and press ENTER",
                                           container=False)

                answer_box = gr.Textbox(lines=5, placeholder="Final Answer", show_label=False, container=False)

                docs_boxes = gr.HTML()

            with gr.Column(elem_id='right-col'):
                knowledge_explains = ["knowledge1: paragraph-level rerank web docs plus wiki items",
                                      "knowledge2: web docs plus wiki items",
                                      "knowledge3: wiki items", 
                                      "knowledge4: summarize paragraph-level rerank web docs",
                                      "knowledge5: paragraph-level rerank web docs&wiki items&llm gen doc"]
                with gr.Column(elem_id='right-col'):
                    knowledge_boxes = []
                    for i in range(5):
                        knowledge_boxes.append(gr.Textbox(f"Textbox {i}", visible=True, label=knowledge_explains[i]))

                with gr.Column(elem_id='right-col'):
                    responses_boxes = []
                    for i in range(5):
                        responses_boxes.append(gr.Textbox(f"Textbox {i}", visible=True,
                                                          label=f"response{i + 1} based on knowledge{i + 1}"))
                    responses_boxes.append(gr.Textbox(f"Textbox query_only", visible=True,
                                                          label=f"query_only"))

        query_box.submit(_wrapper, query_box, [answer_box, docs_boxes] + knowledge_boxes + responses_boxes)

    return demo
