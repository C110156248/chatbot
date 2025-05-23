import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search

def google_search_results(query, num_results=5):
    """使用 Google Search 搜尋相關資訊"""
    try:
        search_results = []
        # 修正搜尋參數，使用正確的參數名稱
        search_params = {
            'num': num_results,  # 修改 'num_results' 為 'num'
            'lang': 'zh',
            'stop': num_results,
            'pause': 2.0,  # 稍微降低速度，避免被封鎖
        }
        
        # 嘗試執行搜尋
        urls = list(search(query, **search_params))
        
        if not urls:
            st.warning("Google 搜尋未返回結果，可能因為速率限制。")
            return []
            
        for url in urls:
            try:
                # 添加 User-Agent 頭，模擬瀏覽器
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # 提取文本並清理
                    text = soup.get_text(separator=' ', strip=True)
                    # 移除多餘空白與特殊符號
                    text = re.sub(r'\s+', ' ', text)
                    # 限制文本長度
                    text = text[:1500] + ("..." if len(text) > 1500 else "")
                    search_results.append({
                        "url": url,
                        "content": text
                    })
            except Exception as e:
                st.warning(f"無法訪問網址 {url}: {str(e)}")
                continue
        
        if not search_results:
            st.warning("無法從搜尋結果中提取內容")
        
        return search_results
    except Exception as e:
        st.error(f"進行 Google 搜尋時發生錯誤: {str(e)}")
        st.info("嘗試切換到備用搜尋引擎...")
        
        # 這裡可以添加備用搜尋方法，比如使用 Bing 或其他 API
        try:
            # 如果沒有備用搜尋方法，至少返回錯誤信息
            return [{
                "url": "無搜尋結果",
                "content": f"搜尋引擎無法返回結果。可能原因：網路連接問題、API 限制或搜尋引擎限制。錯誤信息: {str(e)}"
            }]
        except:
            return []