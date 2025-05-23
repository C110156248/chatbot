import openai
import re
from modules.text_utils import convert_to_traditional

def get_response(messages, base_url="http://localhost:11434", model="deepseek-r1:14b", api_key="ollama"):
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

def init_models(base_url="http://localhost:11434/v1"):
    """初始化生成模型和嵌入模型"""
    from langchain_ollama import OllamaEmbeddings
    
    generation_model = openai.OpenAI(
        base_url=base_url,
        api_key="ollama",
    )
    embedding_model = OllamaEmbeddings(
        model='shaw/dmeta-embedding-zh'
    )
    
    return generation_model, embedding_model