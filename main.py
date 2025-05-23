import streamlit as st
from modules import init_models, process_document, generate_answer

# Streamlit 應用界面
def main():
    st.title("RAG 知識問答助手")
    st.write("上傳文件並提問，助手會從文件中尋找答案，如果找不到會使用網路搜尋")
    
    # 初始化 session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 初始化模型
    generation_model, embedding_model = init_models()
    
    # 側邊欄 - 上傳文件
    with st.sidebar:
        st.header("上傳文件")
        uploaded_file = st.file_uploader("選擇 PDF,TXT,MD 文件", type=['pdf', 'txt', 'md'])
        
        if uploaded_file is not None:
            if st.button("處理文件"):
                with st.spinner("正在處理文件..."):
                    st.session_state.vectorstore = process_document(uploaded_file, embedding_model)
                if st.session_state.vectorstore:
                    st.success(f"文件 '{uploaded_file.name}' 處理完成！")
                    # 重置聊天歷史
                    st.session_state.chat_history = []
        
        if st.button("清除聊天記錄"):
            st.session_state.chat_history = []
    
    # 主視窗 - 聊天介面
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "source" in message and message["source"]:
                st.caption(message["source"])
    
    # 問題輸入
    question = st.chat_input("請輸入您的問題")
    if question:
        # 顯示使用者問題
        with st.chat_message("user"):
            st.write(question)
        
        # 添加到歷史記錄
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                answer, source_type = generate_answer(question, st.session_state.vectorstore, generation_model)
                st.write(answer)
                st.caption(source_type)
        
        # 添加到歷史記錄
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "source": source_type})

if __name__ == "__main__":
    main()