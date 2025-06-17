import openai
import re
from modules.text_utils import convert_to_traditional
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.vectorstores import FAISS
import numpy as np

def get_response(messages, base_url="http://localhost:11434", model="EntropyYue/chatglm3:latest", api_key="ollama"):
    """從大型語言模型獲取回應"""
    llm = openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    response = llm.chat.completions.create(
        model=model,
        messages=messages
    )
    raw_content = response.choices[0].message.content
    cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
    cleaned_content = convert_to_traditional(cleaned_content.strip())
    return cleaned_content.strip()

class TransformersEmbedding:
    """自定義嵌入類，兼容 LangChain 的 FAISS"""
    def __init__(self, model_name="maidalun1020/bce-embedding-base_v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts):
        """嵌入多個文檔"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 使用 CLS token 的嵌入 (bce-embedding-base_v1 使用 CLS 池化)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(embedding[0])
        return embeddings

    def embed_query(self, text):
        """嵌入單個查詢"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0].cpu().numpy()[0]

def init_models(base_url="http://localhost:11434/v1"):
    """初始化生成模型和嵌入模型"""
    generation_model = openai.OpenAI(
        base_url=base_url,
        api_key="ollama",
    )
    embedding_model = TransformersEmbedding(model_name="maidalun1020/bce-embedding-base_v1")
    return generation_model, embedding_model