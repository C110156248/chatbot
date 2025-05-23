import os
import tempfile
import streamlit as st
import traceback
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_document(uploaded_file, embedding_model):
    """處理上傳的文件並建立向量資料庫"""
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 根據文件類型選擇對應的 loader
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif uploaded_file.name.endswith('.md'):
            loader = UnstructuredMarkdownLoader(tmp_path)
        else:
            st.error(f"不支援的文件格式: {uploaded_file.name}")
            os.unlink(tmp_path)
            return None
        
        # 載入文件
        documents = loader.load()
        # 檢查文件內容
        if not documents or len(documents) == 0:
            st.error("文件內容為空或無法解析")
            os.unlink(tmp_path)
            return None
            
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # 創建向量存儲
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        
        # 清理臨時文件
        os.unlink(tmp_path)
        
        return vectorstore
    
    except Exception as e:
        st.error(f"處理文件時發生錯誤: {str(e)}, 類型: {type(e).__name__}") 
        st.error(f"詳細錯誤{traceback.format_exc()}")
        os.unlink(tmp_path)
        return None