import json
import spacy
from collections import Counter

def analyze_sentence_length(question):
    # 按空格分割句子，计算单词数量
    words = question.split()
    return len(words)

def analyze_lexical_complexity(question):
    # 分词并转换为小写
    words = question.lower().split()
    
    # 计算词汇多样性
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words)
    
    # 计算低频词数量（假设低频词为长度大于5的单词）
    low_freq_words = [word for word in words if len(word) > 6]
    low_freq_count = len(low_freq_words)
    
    return lexical_diversity, low_freq_count

# 加载预训练模型
# nlp = spacy.load("en_core_web_sm")

def analyze_syntax_complexity(question):
    # 解析句子
    doc = nlp(question)
    
    # 计算句子中的依存关系数量
    dependency_count = len(list(doc))
    
    # 计算句子中的从句数量（通过连词标记）
    clause_count = sum(1 for token in doc if token.dep_ == "mark")
    
    return dependency_count, clause_count



def analyze_complexity(question):
    # 句法复杂度
    dependency_count, clause_count = analyze_syntax_complexity(question)
    print(f"dependency_count: {dependency_count}")
    print(f"clause_count: {clause_count}")
    # 词汇复杂度
    lexical_diversity, low_freq_count = analyze_lexical_complexity(question)
    print(f"lexical_diversity: {lexical_diversity}")
    print(f"low_freq_count: {low_freq_count}")
    # 综合评分（加权平均）
    complexity_score = (
        0.3 * clause_count +  # 从句数量权重
        0.2 * dependency_count +  # 依存关系权重
        0.3 * low_freq_count  +# 低频词权重
        0.2* lexical_diversity #词汇多样性权重
    )
    
    return complexity_score

# 示例
# question = "What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?"
# score = analyze_complexity(question)
# print(f"综合复杂度评分: {score}")

categories = {
    # 事实检索型：询问具体细节
    "fact_retrieval": [
        "how many",   
        "name the", "identify the", "primary tool", "tools used",
        "key tools", "specific item", "material"
    ],
    # 过程描述型：描述步骤/流程
    "process_description": [
        "describe the process", "steps taken", "sequence of actions",
        "from start to finish", "how to", "method", "progression",
        "workflow", "procedures", "step-by-step", "sequentially"
    ],
    #因果推断
    "causal_reasoning": [
        "explain","infer","deduce","why",
        "why did", "how did", "contribute to", "result in", "because",
        "rationale behind", "led to", "impact of","relationship between"
    ],
        # 主题概括型：总结目标/主题
    "theme_summary": [
        "summarize",
        "overarching theme", "primary objective", "main goal",
        "central purpose", "fundamental intention", "core focus",
        "essential aim", "principal motivation", "underlying narrative"
    ],
        # 比较分析型：对比异同
    "comparative_analysis": [
        "compare", "contrast", "similarities", "differences",
        "distinguish", "relative importance", "more significant",
        "versus", "whereas", "unlike", "analogous"
    ],
        # 行为推断型：推断动机/意义
    "behavior_inference": [
        "infer", "deduce", "possible reason", "underlying motivation",
        "significance of", "implications", "hidden purpose",
        "unspoken intention", "symbolic meaning"
    ],
        # 关键点识别型：识别转折/关键
    "key_moment": [
        "critical step", "turning point", "pivotal moment",
        "decisive action", "crucial stage", "defining event",
        "watershed moment", "breakthrough", "game-changing"
    ],
    # 交互分析型：分析人物互动
    "interaction_analysis": [
        "interaction between", "collaboration", "communication",
        "dynamic with", "relationship with", "coordination",
        "exchange with", "interplay", "cooperation", "conflict"
    ]
        
}

def classify_question(question):

    for category, keywords in categories.items():
        if any(keyword in question.lower() for keyword in keywords):
            return category
    return "other"

def evaluate_question(
  question  
):
    #综合评价一个句子是否需要复杂推理
    if analyze_sentence_length(question=question):
        return True
    #评估输入问题的类别、难度、是否需要复杂推理等
    category = classify_question(question)
    difficulty = assess_difficulty(question)
    
    return False

data_list=[]
question_list=[]
# path="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/outputs/egoschema/32frm_sum_sc_res_0.1_0.5_86.json"
path="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/outputs/videomme/32frm_baseline.json"
with open(path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            # 逐行解析每个 JSON 字典
            data = json.loads(line.strip())  # 去除行尾的换行符等空白字符
            data_list.append(data)
            question_list.append(data['question'])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line}")
            print(f"Error: {e}")

sample="Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?"
evaluate_question(sample)


# 使用字典推导式创建新字典
# category_num = {key: 0 for key in categories.keys()}
# category_num['other']=0

# difficulty_num={}
# difficulty_num['simple']=0
# difficulty_num['medium']=0
# difficulty_num['hard']=0

# for question in question_list:
#     category,difficulty=evaluate_question(question=question)
#     category_num[category]+=1
#     difficulty_num[difficulty]+=1
    
# print(category_num)
# print(difficulty_num)
# # print(f'fact number: {category_num["fact"]}')
# print(f'reasoning number: {category_num["reasoning"]}')
# print(f'summary number: {category_num["summary"]}')
# print(f'other number: {category_num["other"]}')

# print(f'simple number: {difficulty_num["simple"]}')
# print(f'medium number: {difficulty_num["medium"]}')
# print(f'hard number: {difficulty_num["hard"]}')


# with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/videomme_questions.json", "w", encoding="utf-8") as f:
#     json.dump(question_list, f, ensure_ascii=False, indent=4)