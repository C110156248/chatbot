from modules.models import get_response
from modules.search import google_search_results

def generate_answer(question, vectorstore, generation_model):
    """生成問題回答，結合文件資料庫和網路搜尋"""
    # 先嘗試從向量數據庫中檢索相關資訊
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""基於以下資訊回答問題。如果資訊中沒有答案，請明確說明您無法從提供的資訊中找到答案。
            資訊:{context}
            問題: {question}
"""
            messages = [{"role": "user", "content": prompt}]
            answer = get_response(messages, model="deepseek-r1:14b")
            return answer, "從文檔中找到"
            
    # 如果沒有找到相關資訊，使用 Google Search
    search_results = google_search_results(question)
    if search_results:
        search_context = "\n\n".join([f"來源 ({result['url']}): {result['content']}" for result in search_results])
        prompt = f"""基於以下從網路搜尋到的資訊回答問題。如果資訊中沒有答案，請明確說明您無法找到答案。
資訊:
{search_context}
問題: {question}
"""
        messages = [{"role": "user", "content": prompt}]
        answer = get_response(messages)
        return answer, "從網路搜尋中找到"
        
    # 如果都找不到，使用模型直接回答    
    prompt = f"回答以下問題，如果您不確定答案，請誠實地說明: {question}"
    messages = [{"role": "user", "content": prompt}]
    answer = get_response(messages, model="deepseek-r1:14b")
    return answer, "使用模型知識直接回答"