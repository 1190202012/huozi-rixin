import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

zh_prompt = "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\n{user_input}<|endofutterance|>\n<|beginofutterance|>助手\n"

zh_user_input = "请用简要的根据以下文档回答问题。\n\n文档：{doc}\n\n问题：{query}"

zh_examples = [
    {
        "query": "请问2023年爆发的巴以冲突局势如何？",
        "doc": """2023年10月7日，以哈马斯为首的巴勒斯坦武装团体与以色列军队爆发武装冲突。战争始于哈马斯对以色列南部的入侵，以色列军队旋即报复性空袭巴勒斯坦的加沙地带，并在数日后开始准备对加沙的地面入侵。该战争是赎罪日战争以来，数十年来最激烈的一场战争，是第五场发生在加沙地带的战争，也是始于1948年5月的以巴冲突的一部分。战争开始之前的2023年，以巴暴力冲突加剧，造成247名巴勒斯坦人、32名以色列人和2名外国公民丧生。截至10月底，战争已造成至少1,500名以色列人和8,000名巴勒斯坦人死亡，其中包括3,000多名儿童；还有230多名以色列人和外国公民仍被哈马斯劫为人质。\n\n2023年10月7日清晨，哈马斯发动阿克萨洪水行动（阿拉伯语：عملية طوفان الأقصى‎，罗马化：amaliyyat ṭūfān al-Aqṣā），从加沙地带向以色列发射至少3,000馀枚火箭弹，同时约2,500名武装分子突破隔离墙突袭加沙周边的以色列地区，并在邻近的犹太屯垦区屠杀平民，袭击以色列国防军的军事基地。至少44个国家将此次突袭定性为恐怖袭击。哈马斯声称此次袭击是为了回应以色列“亵渎”阿克萨清真寺、对加沙地带的封锁、以色列定居点的扩张和其定居者的暴力行为，以及数十年来以色列对巴勒斯坦人的暴行。\n\n据以色列政府表示，至少1,400名以色列人和外国公民在10月7日的袭击中遇害，其中雷姆音乐节大屠杀即已有260人被杀。至少200名以色列平民和战俘被劫持至加沙作为人质。在清除以色列南部的哈马斯分子之后，以色列国防军旋即发动铁剑行动（希伯来语：מלחמת "חרבות ברזל"‎，罗马化：Mlchmt "chrvvt vrzl"）报复性空袭加沙，并在一天后正式向哈马斯宣战。在冲突开始的前六天，以色列总共投下了6,000枚炸弹。10月27日，联合国大会投票结果显示，绝大多数国家呼吁立即实行人道主义停火。10月28日，当以色列开始对加沙进行地面入侵时，援助机构再次警告加沙正在发生平民的灾难。11月初，持续投弹的威力据估算已累积到相当于广岛原子弹爆炸的1倍多。不过即使受到海量航弹洗礼，哈玛斯靠地道与废墟仍能保存伏击战力，致以军地面部队攻势受阻拉长。\n\n大量平民在战争中遭到杀害，联合国特别报告员小组和人权组织指控以色列和哈马斯均犯有战争罪。10月9日，以色列全面封锁加沙地带，切断对该地食品、水、电和燃料供应，加剧了对人道主义危机的担忧。10月13日，以色列要求加沙北部的110万平民全部撤离至南部，而哈马斯则呼吁居民留在家中，并封锁了通往南部的道路。据联合国报告，约有100万巴勒斯坦人在加沙境内流离失所，占加沙总人口的近一半。另有20多万以色列人在以色列境内流离失所。加沙卫生部表示，截至10月30日，空袭已造成7,000名巴勒斯坦人和50多名联合国近东巴勒斯坦难民救济和工程处工作人员死亡。"""
    },
    {
        "query": "李克强总理逝世的具体情况是？",
        "doc": """李克强同志遗体在京火化\n习近平李强赵乐际王沪宁蔡奇丁薛祥李希韩正等到八宝山革命公墓送别。胡锦涛送花圈表示哀悼\n李克强同志抢救期间和逝世后，习近平李强赵乐际王沪宁蔡奇丁薛祥李希韩正胡锦涛等同志，前往医院看望或通过各种形式对李克强同志逝世表示沉痛哀悼并向其亲属表示深切慰问\n新华社北京11月2日电 中国共产党的优秀党员，久经考验的忠诚的共产主义战士，杰出的无产阶级革命家、政治家，党和国家的卓越领导人，中国共产党第十七届、十八届、十九届中央政治局常委，国务院原总理李克强同志的遗体，2日在北京八宝山革命公墓火化。李克强同志因突发心脏病，经全力抢救无效，于2023年10月27日0时10分在上海逝世，享年68岁。\n李克强同志抢救期间和逝世后，习近平、李强、赵乐际、王沪宁、蔡奇、丁薛祥、李希、韩正、胡锦涛等同志，前往医院看望或通过各种形式对李克强同志逝世表示沉痛哀悼并向其亲属表示深切慰问。\n2日上午，八宝山革命公墓礼堂庄严肃穆，哀乐低回。正厅上方悬挂着黑底白字的横幅“沉痛悼念李克强同志”，横幅下方是李克强同志的遗像。李克强同志的遗体安卧在鲜花翠柏丛中，身上覆盖着鲜红的中国共产党党旗。\n上午9时许，习近平和夫人彭丽媛，李强、赵乐际、王沪宁、蔡奇、丁薛祥、李希、韩正等，在哀乐声中缓步来到李克强同志的遗体前肃立默哀，向李克强同志的遗体三鞠躬，并与李克强同志亲属一一握手，表示慰问。胡锦涛送花圈，对李克强同志逝世表示哀悼。\n党和国家有关领导同志前往送别或以各种方式表示哀悼。中央和国家机关有关部门负责同志，李克强同志生前友好和家乡代表也前往送别。""",
    }
]
zh_examples = {i["query"]:i["doc"] for i in zh_examples}

en_prompt = "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\n{user_input}<|endofutterance|>\n<|beginofutterance|>助手\n"

# en_user_input_1 = "Please briefly answer the question with following documents.\n\nDocuments: {doc}\n\nQuestion：{query}"
en_user_input = "Assuming the following paragraphs are true:\n\n{doc}\n\nPlease directly answer the following question with one or few words:\n{query}"

en_examples = [
    {
        "query": "Who is the last man on the moon?",
        "doc": "Apollo 17 mission commander Eugene Cernan holds the lower corner of the U.S. flag during the mission's first moonwalk on Dec. 12, 1972. Cernan, the last man on the moon, traced his only child's initials in the dust before climbing the ladder of the lunar module the last time."
    },
    {
        "query": "When did the last man on the moon?",
        "doc": "Apollo 17 mission commander Eugene Cernan holds the lower corner of the U.S. flag during the mission's first moonwalk on Dec. 12, 1972. Cernan, the last man on the moon, traced his only child's initials in the dust before climbing the ladder of the lunar module the last time."
    },
    {
        "query": "What is the name of Elon Musk AI?",
        "doc": """On November 3, Musk stated that xAI would debut its first artificial intelligence model to a "select group". On November 4, 2023, Musk and xAI unveiled Grok, an AI chatbot that is heavily integrated with X (formerly Twitter), for its Premium+ subscribers only."""
    }
]
en_examples = {i["query"]:i["doc"] for i in en_examples}

generate_kwargs = {
    "max_new_tokens": 200,
    "do_sample": True,
    # "repetition_penalty": 1.03,
    "top_k": 30,
    "top_p": 0.9,
}

example_list = [zh_prompt.replace("{user_input}", zh_user_input.replace("{query}", q).replace("{doc}", d)) for q, d in zh_examples.items()] + \
[en_prompt.replace("{user_input}", en_user_input.replace("{query}", q).replace("{doc}", d)) for q, d in en_examples.items()]

model = AutoModelForCausalLM.from_pretrained("HIT-SCIR/huozi-7b-rlhf")
tokenizer = AutoTokenizer.from_pretrained("HIT-SCIR/huozi-7b-rlhf")

device = torch.device("cuda:0")
tokenizer.padding_side = "left"
tokens = tokenizer(example_list, padding="longest", return_tensors="pt")
tokens = tokens.to(device)
model.to(device)
outputs = model.generate(**tokens, **generate_kwargs)

raw_result = tokenizer.batch_decode(outputs)

print(raw_result)

result = []
for r in raw_result:
    temp = r.split("<|beginofutterance|>助手\n")[1].split("<|endofutterance|>")[0]
    result.append(temp)

print(result)

# raw_result = [tokenizer.decode(output) for output in outputs]

# print(raw_result)

# print("_" * 40)

# result = []
# for r in raw_result:
#     temp = r.split("<|beginofutterance|>助手\n")[1].split("<|endofutterance|>")[0]
#     result.append(temp)

# print(result)

# 真实推理结果
text = ['巴以冲突是始于1948年的以巴冲突的一部分。自2023年以来，该冲突加剧，造成至少1,500名以色列人和8,000名巴勒斯坦人死亡，其中包括3,000多名儿童；还有230多名以色列人和外国公民仍被哈马斯劫为人质。截至10月底，战争已造成至少1,500名以色列人和8,000名巴勒斯坦人死亡，其中包括3,000多名儿童；还有230多名以色列人和外国公民仍被哈马斯劫为人质。此次冲突是自赎罪日战争以来，数十年来最激烈的一场战争，也是第五场发生在加沙地带的战争。联合国将此次冲突定性为恐怖袭击，绝大多数国家呼吁立即实行人道主义停火。', 
        '李克强同志抢救期间和逝世后，习近平、李强、赵乐际、王沪宁、蔡奇、丁薛祥、李希、韩正、胡锦涛等同志，前往医院看望或通过各种形式对李克强同志逝世表示沉痛哀悼并向其亲属表示深切慰问。李克强同志因突发心脏病，经全力抢救无效，于2023年10月27日0时10分在上海逝世，享年68岁。', 
        'Eugene Cernan',
        '1972',
        'Grok']