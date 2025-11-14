from transformers import AutoTokenizer
from transformers import pipeline
import transformers
import torch
import os
import re
import json
import openai
from tqdm import tqdm
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import warnings 
from transformers import LogitsProcessor
import copy
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
warnings.filterwarnings('ignore')

model_path = '../models/Meta-Llama-3-8B-Instruct'

device_0 = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device_0  
)

from transformers import LogitsProcessor
import torch

class FastLanguageConsistencyProcessor(LogitsProcessor):
    def __init__(self, target_token_ids, neutral_token_ids, penalty_factor=0.7, boost_factor=1.2):
        self.penalty_factor = penalty_factor
        self.boost_factor = boost_factor

        self.target_token_ids = set(target_token_ids)
        self.neutral_token_ids = set(neutral_token_ids)
        self.vocab_size = None
        self.mask = None  

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.mask is None or self.vocab_size != scores.size(-1):
            self.vocab_size = scores.size(-1)
            mask = torch.ones(self.vocab_size, device=scores.device) * self.penalty_factor

            for tid in self.neutral_token_ids:
                mask[tid] = 1.0

            for tid in self.target_token_ids:
                mask[tid] = self.boost_factor

            self.mask = mask

        return scores * self.mask.unsqueeze(0)  # [batch_size, vocab_size]


def build_single_language_prompt(
    sample,
    query_lang="en",
    prompt_lang="zh",
    context_lang="en",
    use_distractor=False,
    num_distractors=1,
    max_evidence=2,
    system_role="scientific_qa",
    user_prompt_template=None,
    use_icl_examples=False,    
    icl_lang="en"               
):
    lang_map = {
        "zh": "Chinese",
        "en": "English",
        "ar": "Arabic",
        "ru": "Russian"
    }

    # === Prompt ===
    default_user_prompts = {
    "zh": (
        "现在开始正式提问：{query}\n\n以下是与问题相关的背景材料：\n\n{context}\n\n"
        "请根据上述内容作答。\n"
        "请按照以下格式严格输出：\n"
        "【推理过程】：...（多步推理）\n"
        "【最终答案】：...（简洁直接的回答）"
    ),
    "en": (
        "Now let's begin the actual question: {query}\n\nHere are some background materials related to the question:\n\n{context}\n\n"
        "Please answer the question directly based on the above information.\n"
        "Use the following format strictly:\n"
        "Reasoning: ... (step-by-step)\n"
        "Final Answer: ... (concise)"
    ),
    "ar": (
        "الآن نبدأ بالسؤال الفعلي: {query}\n\nفيما يلي بعض المواد المرجعية المتعلقة بالسؤال:\n\n{context}\n\n"
        "يرجى الإجابة على السؤال مباشرةً استنادًا إلى المعلومات أعلاه.\n"
        "يرجى اتباع هذا التنسيق في إجابتك:\n"
        "التحليل: ... (خطوات التفكير)\n"
        "الإجابة النهائية: ... (إجابة مختصرة)"
    ),
    "ru": (
        "Теперь переходим к основному вопросу: {query}\n\nВот некоторые материалы, относящиеся к вопросу:\n\n{context}\n\n"
        "Пожалуйста, дайте прямой ответ на вопрос, опираясь на приведённую выше информацию.\n"
        "Просьба использовать следующий формат:\n"
        "Рассуждение: ... (по шагам)\n"
        "Окончательный ответ: ... (краткий ответ)"
    )
}

    user_prompt_template = user_prompt_template or default_user_prompts.get(prompt_lang)

    # === system prompt ===
    system_prompts = {
        "scientific_qa": {
            "zh": "你是一个严谨的科学问答助手，善于根据给定文档回答复杂问题。请用中文回答问题。",
            "en": "You are a rigorous scientific question answering assistant. Please answer the question in English based on the given documents.",
            "ar": "أنت مساعد موثوق للإجابة على الأسئلة العلمية. أجب عن السؤال باللغة العربية بناءً على النصوص المقدمة.",
            "ru": "Вы — надежный помощник по научным вопросам. Пожалуйста, ответьте на вопрос на русском языке, используя приведённые документы."
        }
    }
    system_prompt = system_prompts[system_role][prompt_lang]

    icl_examples_multilang = {
        "en": [
            ("Below are some example question and answer pairs (with reasoning):", ""),
            (
                "Question: What is the primary duty of the Austrian Pinscher and the Maremma Sheepdog?",
                "Reasoning: Both breeds were historically used to guard livestock and property.\nFinal Answer: to guard"
            ),
            (
                "Question: Which is a type of gun dog, the Labrador Retriever or the Tibetan Terrier?",
                "Reasoning: The Labrador was bred for retrieving game, unlike the Tibetan Terrier.\nFinal Answer: Labrador Retriever"
            ),
            (
                "Question: Do the bands named Phoenix and Shearwater specialize in the exact same genre?",
                "Reasoning: Phoenix is indie pop; Shearwater is more atmospheric indie rock.\nFinal Answer: no"
            ),
            (
                "Question: Were the films Tonka and 101 Dalmatians released in the same decade?",
                "Reasoning: Tonka was released in 1958 and 101 Dalmatians in 1996, which are different decades.\nFinal Answer: no"
            )
        ],
        "zh": [
            ("下面是一些带有推理过程的问题和答案示例：", ""),
            (
                "问题：奥地利平犬和马雷马牧羊犬的主要职责是什么？",
                "推理过程：这两种狗历史上都被用来保护家畜和财产。\n最终答案：守卫"
            ),
            (
                "问题：拉布拉多寻回犬和西藏梗哪一种是猎枪犬？",
                "推理过程：拉布拉多是为了捡回猎物而培育的，而西藏梗不是。\n最终答案：拉布拉多寻回犬"
            ),
            (
                "问题：乐队Phoenix和Shearwater属于完全相同的音乐类型吗？",
                "推理过程：Phoenix是独立流行风格，而Shearwater更偏向氛围感的独立摇滚。\n最终答案：不是"
            ),
            (
                "问题：电影《托卡》和《101忠狗》是否在同一个十年上映？",
                "推理过程：《托卡》是1958年上映，《101忠狗》是1996年上映，属于不同十年。\n最终答案：不是"
            )
        ],
        "ru": [
            ("Ниже приведены примеры вопросов и ответов с рассуждением:", ""),
            (
                "Вопрос: Какова основная обязанность австрийского пинчера и мареммской овчарки?",
                "Рассуждение: Обе породы исторически использовались для охраны скота и имущества.\nОкончательный ответ: охрана"
            ),
            (
                "Вопрос: Какая из пород — лабрадор ретривер или тибетский терьер — относится к охотничьим собакам?",
                "Рассуждение: Лабрадор был выведен для поиска и подачи дичи, в отличие от тибетского терьера.\nОкончательный ответ: лабрадор ретривер"
            ),
            (
                "Вопрос: Играют ли группы Phoenix и Shearwater в одном и том же музыкальном жанре?",
                "Рассуждение: Phoenix — инди-поп, Shearwater — атмосферный инди-рок.\nОкончательный ответ: нет"
            ),
            (
                "Вопрос: Были ли фильмы Tonka и 101 далматинец выпущены в одном десятилетии?",
                "Рассуждение: Tonka вышел в 1958 году, а 101 далматинец — в 1996, это разные десятилетия.\nОкончательный ответ: нет"
            )
        ],
        "ar": [
            ("فيما يلي بعض أمثلة الأسئلة والإجابات مع التفكير:", ""),
            (
                "السؤال: ما هي المهمة الأساسية لكلب أوسترين بنشر وكلب ماريمّا شيب دوغ؟",
                "التحليل: استخدمت السلالتان تاريخيًا لحراسة الماشية والممتلكات.\nالإجابة النهائية: الحراسة"
            ),
            (
                "السؤال: أي من الكلبين هو كلب صيد: لابرادور ريتريفر أم التيرير التبتي؟",
                "التحليل: تم تربية لابرادور لاسترجاع الطرائد، على عكس التيرير التبتي.\nالإجابة النهائية: لابرادور ريتريفر"
            ),
            (
                "السؤال: هل تتخصص الفرق الموسيقية Phoenix وShearwater في نفس النوع الموسيقي تمامًا؟",
                "التحليل: Phoenix هي فرقة بوب مستقلة، بينما Shearwater أكثر ميلًا إلى الروك الجوي.\nالإجابة النهائية: لا"
            ),
            (
                "السؤال: هل تم إصدار الفيلمين Tonka و101 Dalmatians في نفس العقد؟",
                "التحليل: تم إصدار Tonka في 1958 و101 Dalmatians في 1996، وهما عقدان مختلفان.\nالإجابة النهائية: لا"
            )
        ]
    }


    query = sample["query"][query_lang]
    evidences = sample["evidence"].get(context_lang, [])[:max_evidence]
    distractors = sample["other related text"].get(context_lang, [])[:num_distractors] if use_distractor else []
    context_text = "\n\n".join([f"{i+1}. {para}" for i, para in enumerate(evidences + distractors)])

    icl_text = ""
    if use_icl_examples:
        examples = icl_examples_multilang.get(icl_lang, icl_examples_multilang)
        icl_text = "\n\n".join([f"{q}\n {a}" for q, a in examples]) + "\n\n"

    query_with_context = user_prompt_template.format(query=query, context=context_text)
    user_prompt = icl_text + query_with_context

    full_prompt = (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    return {
        "system": system_prompt,
        "user": user_prompt,
        "full_prompt": full_prompt,
        "query_lang": query_lang,
        "prompt_lang": prompt_lang,
        "context_lang": context_lang,
        "evidence_count": len(evidences),
        "distractor_count": len(distractors),
        "icl_used": use_icl_examples,
        "icl_lang": icl_lang
    }


def get_output(text, infer_type, processor):

    if infer_type == 'base':
        sequences = pipeline(
            text,
        #     do_sample=True,
            top_k=10,
        #     num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
            temperature=0.8,
        #     max_length=200,
        )

        return sequences[0]['generated_text'].split('<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n')[1]
    elif  infer_type == 'constrained':

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        output_ids = model.generate(
      
            
            input_ids=inputs["input_ids"],
            max_new_tokens=500,
            do_sample=True,         
            temperature=0.8,        
            top_k=100,               
            top_p=0.95,             
            # repetition_penalty=1.2,
            logits_processor=[processor],
        )
        
        sequences = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        return sequences.split('<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n')[1]

import re

def extract_reason_and_answer_multilang(text, lang="en"):

    lang_prefixes = {
        "en": {
            "reason_prefix": r"(?:Reasoning|Explanation|Steps?):?\s*",
            "answer_prefix": r"(?:Final Answer|Answer):?\s*"
        },
        "zh": {
            "reason_prefix": r"(?:推理过程|思考过程|【?推理过程】?|【?思考过程】?)[:：]?\s*",
            "answer_prefix": r"(?:最终答案|答案|【?最终答案】?|【?答案】?)[:：]?\s*"
        },
        "ru": {
            "reason_prefix": r"(?:Рассуждение|Обоснование|Объяснение):?\s*",
            "answer_prefix": r"(?:Окончательный ответ|Ответ):?\s*"
        },
        "ar": {
            "reason_prefix": r"(?:التحليل|الشرح|التفسير):?\s*",
            "answer_prefix": r"(?:الإجابة النهائية|الجواب النهائي|الإجابة):?\s*"
        }
    }

    def try_extract_with_prefixes(text, reason_prefix, answer_prefix):
        reasoning, answer = "", ""


        reason_match = re.search(reason_prefix, text, re.IGNORECASE)
        if reason_match:
            r_start = reason_match.end()
            reasoning = text[r_start:].strip()


            answer_in_reason = re.search(answer_prefix, reasoning, re.IGNORECASE)
            if answer_in_reason:
                reasoning = reasoning[:answer_in_reason.start()].strip()


        answer_matches = list(re.finditer(answer_prefix, text, re.IGNORECASE))
        if answer_matches:
            a_start = answer_matches[-1].end()
            answer = text[a_start:]
            answer = re.split(r"<\|eot_id\|>|\n", answer)[0].strip()

        return reasoning, answer



    prefixes = lang_prefixes.get(lang)
    reasoning, answer = try_extract_with_prefixes(
        text,
        prefixes["reason_prefix"],
        prefixes["answer_prefix"]
    )

    if not reasoning and not answer:
        for alt_lang , prefixes in lang_prefixes.items():
            if alt_lang  == lang:
                continue
            reasoning, answer = try_extract_with_prefixes(
                text,
                prefixes["reason_prefix"],
                prefixes["answer_prefix"]
            )
            if reasoning or answer:
                break  


    if not answer:

        candidate_sents = re.split(r"[。！？!?\.]", reasoning.strip())
        candidate_sents = [s.strip() for s in candidate_sents if len(s.strip()) > 2]

        if candidate_sents:
            try:
                answer = candidate_sents[-2]
            except:
                try:
                    answer = candidate_sents[-1]
                except:
                    answer = ''
    
    answer = answer.replace("<|eot_id|>", "")
    return reasoning.strip(), answer.strip()


def get_prediction(
    data, 
    query_lang="en",
    prompt_lang="zh",
    context_lang="en",
    use_distractor=False,
    num_distractors=0,
    max_evidence=2,
    system_role="scientific_qa",
    use_icl_examples=True,    
    icl_lang="zh",
    infer_type = 'base'):

    output = []

    for i in tqdm(range(len(data))):
        prediction = dict()
        prediction['query_lang'] = [query_lang]
        prediction['prompt_lang'] = [prompt_lang]
        prediction['context_lang'] = [context_lang]
        prediction['num_distractors'] = [num_distractors]
        prediction['label'] = []
        prediction['cot'] = []
        prediction['pred'] = []
        
        temp_prompt = build_single_language_prompt(
                    data[i],
                    query_lang=query_lang,
                    prompt_lang=prompt_lang,
                    context_lang=context_lang,
                    use_distractor=use_distractor,
                    num_distractors=num_distractors,
                    max_evidence=max_evidence,
                    system_role=system_role,
                    user_prompt_template=None,
                    use_icl_examples=use_icl_examples,    
                    icl_lang=icl_lang  
                )
        sequences = get_output(temp_prompt["full_prompt"], infer_type = infer_type, processor = processor)
        prediction['input'] = temp_prompt["full_prompt"]
        prediction['output'] = sequences
        sequences = extract_reason_and_answer_multilang(sequences, lang=query_lang)
        prediction['label'].append(data[i]['answer'][query_lang])
        try:
            prediction['cot'].append(sequences[0])
        except:
            print(i)
        try:
            prediction['pred'].append(sequences[1].split('<|eot_id|>')[0])
        except:
            print(i)
        output.append(prediction)
        
    return output


def calculate_bleu(reference, prediction, tokenizer):

    reference_tokens = [tokenizer.tokenize(reference)]
    prediction_tokens = tokenizer.tokenize(prediction)
    
    bleu_score_1 = sentence_bleu(reference_tokens, prediction_tokens)
    bleu_score_2 = sentence_bleu(reference_tokens, prediction_tokens, weights=(1/2,) * 2)
    bleu_score_3 = sentence_bleu(reference_tokens, prediction_tokens, weights=(1/3,) * 3)
    return bleu_score_1, bleu_score_2, bleu_score_3

def calculate_rouge(reference, prediction, tokenizer):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=tokenizer)
    
    scores = scorer.score(reference, prediction)
    return scores['rouge1'][-1], scores['rouge2'][-1], scores['rougeL'][-1]

def evaluate(reference, prediction, tokenizer):
    bleu_score = calculate_bleu(reference, prediction, tokenizer)
    
    rouge_scores = calculate_rouge(reference, prediction, tokenizer)
    
    return bleu_score, rouge_scores

def get_auto_score(data):

    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    rouge_1 = []
    rouge_2 = []
    rouge_L = []
    
    for i in tqdm(range(len(data))):
        reference = data[i]['label'][0]
        prediction = data[i]['pred'][0]
        bleu_score, rouge_score = evaluate(reference, prediction, tokenizer)
 
        data[i]['bleu_1'] = bleu_score[0]
        data[i]['bleu_2'] = bleu_score[1]
        data[i]['bleu_3'] = bleu_score[2]

        data[i]['rouge_1'] = rouge_score[0]
        data[i]['rouge_2'] = rouge_score[1]
        data[i]['rouge_L'] = rouge_score[2]

        bleu_1.append(bleu_score[0])
        bleu_2.append(bleu_score[1])
        bleu_3.append(bleu_score[2])
        rouge_1.append(rouge_score[0])
        rouge_2.append(rouge_score[1])
        rouge_L.append(rouge_score[2])

    print(f"{'BLEU'}: bleu_1={np.mean(bleu_1):.4f}, bleu_2={np.mean(bleu_2):.4f}, bleu_3={np.mean(bleu_3):.4f}")
    print(f"{'ROUGE'}: rouge_1={np.mean(rouge_1):.4f}, rouge_2={np.mean(rouge_2):.4f}, rouge_L={np.mean(rouge_L):.4f}")
    return data



def get_language_token_ids(tokenizer, target_lang="zh", vocab_size_limit=5000000):

    unicode_ranges = {
        "zh": ('\u4e00', '\u9fff'),
        "ar": ('\u0600', '\u06FF'),
        "ru": ('\u0400', '\u04FF'),
        "en": ('\u0000', '\u007F'),
    }

    low, high = unicode_ranges.get(target_lang, (None, None))
    if low is None:
        raise ValueError(f"Unsupported language: {target_lang}")

    def is_target_language(text, low, high):
        return any(low <= ch <= high or ch.isdigit() for ch in text)


    language_token_ids = set()
    for token_id in range(min(tokenizer.vocab_size, vocab_size_limit)):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if is_target_language(decoded, low, high):
                language_token_ids.add(token_id)
                continue
            fallback_decoded = tokenizer.decode(token_id)
            if is_target_language(fallback_decoded, low, high):
                language_token_ids.add(token_id)
        except Exception:
            continue


    special_token_dict = tokenizer.special_tokens_map
    special_token_ids = {
        tokenizer.convert_tokens_to_ids(tok)
        for tok in special_token_dict.values() if tok
    }


    whitelist_chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', ',', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}',
        '-', '/', '\\', '@', '#', '%', '&', '*', '=', '+', '_',
        '。', '，', '？', '！', '；', '：', '“', '”', '‘', '’', '（', '）', '《', '》', '、',
        ' ', '\u3000', '\t'
    ]

    whitelist_token_ids = set()
    for ch in whitelist_chars:
        try:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            whitelist_token_ids.update(ids)
        except Exception:
            continue

    neutral_token_ids = special_token_ids.union(whitelist_token_ids)

    return list(language_token_ids), list(neutral_token_ids)


def is_expected_language_present(text, target_lang):

    lang_patterns = {
        "zh": r"[\u4e00-\u9fff0-9]",
        "en": r"[a-zA-Z0-9]",
        "ru": r"[А-Яа-яЁё0-9]",
        "ar": r"[\u0600-\u06FF0-9]"
    }

    pattern = lang_patterns.get(target_lang)
    if not pattern:
        return False

    return re.search(pattern, text) is not None

    def get_some_ids(data, lang):

    id_zero_performance = []
    id_not_target = []
    
    for i in range(len(data)):
        data[i]['id'] = i
        if data[i]['avg_bleu'] <= 1e-5:
            id_zero_performance.append(i)
        if not is_expected_language_present(data[i]['pred'][0], lang):
            id_not_target.append(i)

    return id_zero_performance, id_not_target

def get_avg_score(data):
    for i in range(len(data)):
        data[i]['avg_bleu'] = float(np.mean([data[i]['bleu_1'], data[i]['bleu_2'], data[i]['bleu_3']]))
        data[i]['avg_rouge'] = float(np.mean([data[i]['rouge_1'], data[i]['rouge_2'], data[i]['rouge_L']]))
    return data

target_token_ids_zh, neutral_token_ids = get_language_token_ids(tokenizer, target_lang="zh")
target_token_ids_ar, neutral_token_ids = get_language_token_ids(tokenizer, target_lang="ar")
target_token_ids_ru, neutral_token_ids = get_language_token_ids(tokenizer, target_lang="ru")

processor_zh = FastLanguageConsistencyProcessor(
    target_token_ids=target_token_ids_zh,
    neutral_token_ids = neutral_token_ids,
    penalty_factor=0.8,
    boost_factor=1.2
)

processor_ar = FastLanguageConsistencyProcessor(
    target_token_ids=target_token_ids_ar,
    neutral_token_ids = neutral_token_ids,
    penalty_factor=0.9,
    boost_factor=1.1
)

processor_ru = FastLanguageConsistencyProcessor(
    target_token_ids=target_token_ids_ru,
    neutral_token_ids = neutral_token_ids,
    penalty_factor=0.8,
    boost_factor=1.2
)

processors = {
            "zh": processor_zh,
            "ar": processor_ar,
            "ru": processor_ru
        }


data_path_1 = "../Result/Multilangual(context)/multi_with_constraint_zh_en_with_RAG_with_ICL.json"
data_path_2 = "../Result/Multilangual(context)/multi_with_constraint_zh_ar_with_RAG_with_ICL.json"
data_path_3 = "../Result/Multilangual(context)/multi_with_constraint_zh_ru_with_RAG_with_ICL.json"

with open(data_path_1, "r", encoding="utf-8") as f:
    zh_en = json.load(f)
with open(data_path_2, "r", encoding="utf-8") as f:
    zh_ar = json.load(f)
with open(data_path_3, "r", encoding="utf-8") as f:
    zh_ru = json.load(f)

processor = processors.get('zh')
zh_en_scd_output = get_prediction(
    zh_en, 
    query_lang="zh",
    prompt_lang="zh",
    context_lang="en",
    use_distractor=False,
    num_distractors=0,
    max_evidence=2,
    system_role="scientific_qa",
    use_icl_examples=True,    
    icl_lang="zh",
    infer_type = 'constrained')

zh_en_scd_output = get_auto_score(zh_en_scd_output)
zh_en_scd_output = get_avg_score(zh_en_scd_output)

output_filename = "../Result/SoftConstrainedDecoding/llama3-8b/hotpotqa/zh_en_scd_prediction.json"
with open(output_filename, "w", encoding="utf-8") as f_out:
    json.dump(zh_en_scd_output, f_out, indent=2, ensure_ascii=False)