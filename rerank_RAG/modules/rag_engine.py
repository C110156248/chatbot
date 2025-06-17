from modules.models import get_response
from modules.search import google_search_results
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化 reranker 模型和 tokenizer（在模組層級載入以避免重複初始化）
tokenizer = AutoTokenizer.from_pretrained("maidalun1020/bce-reranker-base_v1")
model = AutoModelForSequenceClassification.from_pretrained("maidalun1020/bce-reranker-base_v1")
model.eval()  # 設置為評估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def rerank_documents(question, documents, top_n=3):
    ranked_docs = []
    
    for doc in documents:
        # 準備輸入：將問題和文檔內容組合
        input_text = f"{question} [SEP] {doc.page_content}"
        try:
            # 編碼輸入
            inputs = tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # 獲取模型輸出
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.sigmoid(outputs.logits).item()  # 將 logits 轉換為 0-1 分數
            
            ranked_docs.append((doc, score))
        except Exception as e:
            st.warning(f"Error ranking document: {e}")
            ranked_docs.append((doc, 0))
    # 按分數排序並選取前 top_n 個
    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)[:top_n]
    return [doc for doc, score in ranked_docs]

def generate_answer(question, vectorstore, generation_model):
    """生成問題回答，優先從文件資料庫尋找，找不到再進行網路搜尋"""
    # 判斷是否需要外部資源
    evaluation_prompt = f"""
    判斷以下問題是否需要外部資源（如文檔或網路搜尋）來回答。
    如果問題是簡單的問候或常見問題，回答 "否"，否則回答 "是"。
    問題: {question}
    """
    messages = [{"role": "user", "content": evaluation_prompt}]
    evaluation = get_response(messages, model="deepseek-r1:7b").strip().lower()
    print(evaluation)
    if "否" in evaluation:
        prompt = f"請簡短回應: {question}"
        messages = [{"role": "user", "content": prompt}]
        answer = get_response(messages, model="EntropyYue/chatglm3:latest")
        return answer, "直接回答"
    
    doc_context = ""
    web_context = ""
    source_info = []
    
    # 從向量數據庫中檢索相關資訊
    found_in_docs = False
    if vectorstore:
        with st.spinner("從文檔中搜尋相關資訊..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)  # 使用 invoke 替代 get_relevant_documents
            
            if docs:
                # 使用自定義 reranker
                docs = rerank_documents(question, docs, top_n=3)
                doc_context = "\n\n".join([doc.page_content for doc in docs])
                source_info.append("文檔資料庫")
                
                # 檢查文檔是否包含足夠相關的資訊
                evaluation_prompt = f"""
                評估以下資訊是否足夠回答問題。
                只回答 "是" 或 "否"。

                資訊:
                {doc_context}

                問題: {question}
                """
                messages = [{"role": "user", "content": evaluation_prompt}]
                evaluation = get_response(messages, model="deepseek-r1:7b").strip().lower()
                
                if "是" in evaluation:
                    found_in_docs = True
                    st.info("在文檔中找到相關資訊")
    
    # 只有在文檔中找不到充分資訊時，才進行網路搜尋
#     if not found_in_docs:
#         with st.spinner("在文檔中未找到足夠資訊，正在進行網路搜尋..."):
#             search_results = google_search_results(question)
#             if search_results:
#                 web_context = "\n\n".join([f"來源 ({result['url']}): {result['content']}" for result in search_results])
#                 source_info.append("網路搜尋")
    
#     # 根據獲取的資訊選擇如何回答
#     if doc_context or web_context:
#         # 結合文檔和網路搜尋的資訊
#         combined_context = ""
#         if doc_context:
#             combined_context += f"文檔資料:\n{doc_context}\n\n"
#         if web_context:
#             combined_context += f"網路資料:\n{web_context}"
            
#         prompt = f"""基於以下資訊回答問題。如果資訊中沒有完整答案，請考慮所有提供的資訊來提供最全面的回答。
# 資訊:
# {combined_context}

# 問題: {question}
# """
#         messages = [{"role": "user", "content": prompt}]
#         answer = get_response(messages, model="EntropyYue/chatglm3:latest")
#         source_text = "、".join(source_info)
#         return answer, f"從{source_text}中找到"
    
    # 如果都找不到，使用模型直接回答    
    prompt = f"回答以下問題，如果您不確定答案，請誠實地說明: {question}"
    messages = [{"role": "user", "content": prompt}]
    answer = get_response(messages, model="EntropyYue/chatglm3:latest")
    return answer, "使用模型知識直接回答"