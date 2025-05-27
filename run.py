import os
import re
import random
import base64
from io import BytesIO
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from openai import OpenAI
from datasets import load_dataset

from config_private import http_proxy, https_proxy, MODEL2URL, MODEL2KEY, MODEL2MODEL

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def encode_image(image_path):
    buffered = BytesIO()
    image_path.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_openai_api_image(llm_api, image_path, input_text, dataset_name):
    base64_image = encode_image(image_path)
    image_value = f"data:image/png;base64,{base64_image}"
    attempts = 0
    while attempts <= 5:
        try:
            if dataset_name == 'planning_travel':
                response = llm_api.chat.completions.create(
                    model=MODEL2MODEL[args.model_name],
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": input_text}]}],
                    max_tokens=10,
                    temperature=0)
            else:
                response = llm_api.chat.completions.create(
                    model=MODEL2MODEL[args.model_name],
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": input_text, },
                                                     {"type": "image_url",
                                                      "image_url": {"url": image_value,
                                                                    }, }, ], }],
                    max_tokens=10,
                    temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            attempts = attempts + 1
            print(f"{args.model_name} Error occurred on attempt {attempts}: {str(e)}")
    return None


def process_reward(result):
    if result == None:
        result = ""
    result = result.lower()
    result = result.replace(" ", "")
    result = result.replace("\n", "")
    result = result.replace("answer", "")
    result = result.replace(":", "")
    result = result.replace("is", "")
    result = result.replace("the", "")
    result = result.replace("correct", "")
    result = result.replace(".", "")
    result = result.replace("*", "")
    result = result.replace("option", "")
    result = re.sub(r'\D', '', result)
    return result


def eval():
    dataset = load_dataset("MultimodalAgent/Agent-RewardBench")
    result = {}
    for dataset_name, dataset_list in dataset.items():
        print(dataset_name)
        acc_list = []
        result_string_list = []
        result[dataset_name + "_result"] = {}
        for sample in tqdm(dataset_list):
            result_1_raw = call_openai_api_image(llm_api, sample['image'], sample['reward_input_1'], dataset_name)
            result_2_raw = call_openai_api_image(llm_api, sample['image'], sample['reward_input_2'], dataset_name)
            result_1 = process_reward(result_1_raw)
            result_2 = process_reward(result_2_raw)
            acc_list.append([int(result_1 == "1"), int(result_2 == "2")])
            result_string_list.append([result_1_raw, result_2_raw])
        result[dataset_name + "_result"]["acc_list"] = acc_list
        result[dataset_name + "_result"]["result_string_list"] = result_string_list
    with open('experiment/{}.pkl'.format(args.model_name), 'wb') as f:
        pickle.dump(result, f)
    print("end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2-VL-7B-Instruct",
                        choices=["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
                                 "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "gemini-1.5-flash-latest",
                                 "Qwen2-VL-72B-Instruct", "Llama-3.2-90B-Vision-Instruct",
                                 "Qwen2-VL-7B-Instruct", "Llama-3.2-11B-Vision-Instruct",
                                 "Pixtral-12B-2409", "llava-onevision-qwen2-7b-ov-hf",
                                 "Phi-3.5-vision-instruct", "InternVL2-8B",
                                 "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21"])
    args = parser.parse_args()

    random.seed(123)
    np.random.seed(123)

    llm_api = OpenAI(
        api_key=MODEL2KEY[args.model_name],
        base_url=MODEL2URL[args.model_name],
    )
    print(args.model_name)
    eval()
    print("end")
