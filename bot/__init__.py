from roformer import RoFormerTokenizer, RoFormerForCausalLM
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch

# 定义一个Bot
def _compute_sim_score(v1: np.ndarray, v2: np.ndarray) -> float:
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _generate_prompt(question: str, relevant_chunks: List[str]):
    prompt = f'根据文档内容来回答问题，问题是"{question}"，文档内容如下：\n'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt


class QaBot(object):

    # 初始化函数，用于初始化Bot,并接收参数
    def __init__(self, **kwargs):
        pretrained_model = "junnyu/roformer_chinese_sim_char_small"
        self.tokenizer = RoFormerTokenizer.from_pretrained(pretrained_model)
        self.embedding = RoFormerForCausalLM.from_pretrained(pretrained_model)
        # 加载文档，预先计算每个chunk的embedding
        self.doc_path = kwargs['doc_path']
        self.chunks, self.index = self._vec2idx(self.doc_path)
        # 如果参数中有model为chatglm，则获取参数url的值；
        # 如果model为chatgpt，则获取参数key的值；
        # 否则返回None
        self.model = kwargs['model']
        if 'model' in kwargs and kwargs['model'] == 'chatgpt':
            self.chatgpt_api_key = kwargs['key']
        elif 'model' in kwargs and kwargs['model'] == 'chatglm':
            self.chatglm_api_url = kwargs['url']
        else:
            self.model = None

    def _vec2idx(self,path:str):
        chunks = []
        file = open(path)
        for line in file:
            chunks.append(line.strip())
        file.close()
        index = []
        for i in tqdm(range(len(chunks)), desc='计算chunks的embedding'):
            index.append(self._encode_text(chunks[i]))
        return chunks, index

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=384)
        with torch.no_grad():
            outputs = self.embedding.forward(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()[0]
        return embedding

    # 用于处理用户的输入，返回一个字符串
    def ask(self, question):
        # 计算question的embedding
        query_embedding = self._encode_text(question)
        # 根据question的embedding，找到最相关的3个chunk
        relevant_chunks = self._search_index(query_embedding, topk=3)
        # 根据question和最相关的3个chunk，构造prompt
        prompt = _generate_prompt(question, relevant_chunks)
        if self.model == 'chatgpt':
            return self._ask_chatgpt(prompt)
        elif self.model == 'chatglm':
            return self._ask_chatglm(prompt)
        else:
            # 如果没有指定model，则报错提示没有找到model
            raise Exception('没有找到model')

    def _search_index(self, query_embedding: np.ndarray, topk: int = 1) -> List[str]:
        sim_socres = [(i, _compute_sim_score(query_embedding, chunk_embedding))
                      for i, chunk_embedding in enumerate(self.index)]
        sim_socres.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = []
        for i, _ in sim_socres[:topk]:
            relevant_chunks.append(self.chunks[i])
        return relevant_chunks

    def _ask_chatgpt(self, prompt):
        import openai
        openai.api_key = self.chatgpt_api_key
        # 使用chatgpt的api获取答案
        openai.api_key = self.chatgpt_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
        )
        return response.choices[0].message.content.strip()

    def _ask_chatglm(self, prompt):
        import requests

        # 使用chatglm的api获取答案
        resp = requests.post(self.chatglm_api_url, json={
            'prompt': prompt,
            'history': []
        })
        return resp.json()['response']
