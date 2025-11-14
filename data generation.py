import json
import os
import numpy as np
import pandas as pd
import random
import tqdm as tqdm
import openai

path = '../HotpotQA/hotpotqa.json'
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

n_related_text = 3

client = openai.OpenAI(
    #base_url = ,
    #api_key = 
)


def translate_text(text, target_lang):

    lang_map = {
        "zh": "Simplified Chinese",
        "ru": "Russian",
        "ar": "Arabic",
        "es": "Spanish",
        "fr": "French"
    }

    lang_name = lang_map.get(target_lang)

    
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional translation assistant. "
                f"Your task is to translate the given English text into {lang_name}. "
                "Be precise and faithful. Do not add or remove information. "
                "Retain proper names, dates, and technical terms."       ) },
        {
            "role": "user",
            "content": f"Translate the following text:\n{text}"
        }
    ]
    response = client.chat.completions.create(,
            model='gpt-4o',
            messages=messages, temperature = 1.0, max_tokens=300
        )
    return response.choices[0].message.content.strip()

def translate_sample(sample, target_lang):
    translated = dict(sample)  # copy

    translated["query_" + target_lang] = translate_text(sample["query"], target_lang)
    translated["answer_" + target_lang] = translate_text(sample["answer"], target_lang)

    if "evidence" in sample:
        translated["evidence_" + target_lang] = [
            translate_text(p, target_lang) for p in sample["evidence"]
        ]

    if "other related text" in sample:
        translated["other related text_" + target_lang] = [
            translate_text(p, target_lang) for p in sample["other related text"]
        ]

    return translated

def restructure_multilingual_sample(sample, langs=("en", "zh", "ru", "ar")):
    new_sample = {}

    new_sample["_id"] = sample["_id"]
    new_sample["type"] = sample["type"]
    new_sample["level"] = sample["level"]

    #  query / answer
    new_sample["query"] = {lang: sample.get(f"query_{lang}", None) if lang != "en" else sample["query"] for lang in langs}
    new_sample["answer"] = {lang: sample.get(f"answer_{lang}", None) if lang != "en" else sample["answer"] for lang in langs}

    #  evidence / other related 
    for field in ["evidence", "other related text"]:
        new_sample[field] = {}
        for lang in langs:
            if lang == "en":
                new_sample[field][lang] = sample.get(field, [])
            else:
                key = f"{field}_{lang}"
                new_sample[field][lang] = sample.get(key, [])

    return new_sample

data_new = []

for i in tqdm.tqdm(range(len(data))):
    try:
        text = data[i]
        text['other related text'] = text['other related text'][:n_related_text]
        translated_sample = translate_sample(text, target_lang = 'zh')
        translated_sample = translate_sample(translated_sample, target_lang = 'ru')
        translated_sample = translate_sample(translated_sample, target_lang = 'ar')
        translated_sample = restructure_multilingual_sample(translated_sample)
        data_new.append(translated_sample)
    except:
        print(i)
    output_filename = "../HotpotQA/Multilingual_Version/hotpotqa_multi.json"
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(data_new, f_out, indent=2, ensure_ascii=False)