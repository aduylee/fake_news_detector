from flask import Flask, render_template, request, redirect, url_for
import joblib
import re
from newspaper import Article, Config
import tldextract
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load mô hình AI và TF-IDF Vectorizer từ thư mục gốc
model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Danh sách tên miền nguồn tin uy tín (bao gồm cả báo Việt Nam và Quốc tế)
TRUSTED_DOMAINS = [
    'vnexpress.net', 'tuoitre.vn', 'dantri.com.vn', 'thanhnien.vn', 
    'vtv.vn', 'chinhphu.vn', 'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com'
]

def init_db():
    """ Khởi tạo cơ sở dữ liệu SQLite lưu lịch sử kiểm tra """
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            domain TEXT,
            result_label TEXT,
            confidence REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Gọi hàm khởi tạo DB khi chạy app
init_db()

def clean_multilingual_text(text):
    """ Hàm làm sạch văn bản hỗ trợ cả tiếng Anh và tiếng Việt có dấu """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Giữ lại bảng chữ cái, ký tự tiếng Việt có dấu, số và khoảng trắng
    text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừừửữựỳýỷỹỵđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_article_info(url):
    """ 
    Cào thông tin bài báo từ URL (Hỗ trợ cả báo Việt Nam lẫn báo nước ngoài). 
    Sử dụng User-Agent giả lập trình duyệt để chống chặn bot (Anti-bot / 403 Forbidden). 
    """
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10

        article = Article(url, config=config)
        article.download()
        article.parse()

        ext = tldextract.extract(url)
        domain_name = f"{ext.domain}.{ext.suffix}"

        meta_data = {
            'title': article.title if article.title else "Không xác định",
            'authors': ", ".join(article.authors) if article.authors else "Không rõ tác giả",
            'publish_date': article.publish_date.strftime("%d/%m/%Y") if article.publish_date else "Không có ngày đăng",
            'domain': domain_name,
            'text': article.text
        }
        return meta_data
    except Exception as e:
        print(f"Lỗi cào URL báo nước ngoài/trong nước: {e}")
        return None

def get_top_keywords(cleaned_text, top_n=6):
    """ Trích xuất các từ khóa đặc trưng nhất từ văn bản dựa trên vector TF-IDF """
    try:
        vector = tfidf.transform([cleaned_text])
        feature_names = np.array(tfidf.get_feature_names_out())
        sorted_coef_indexes = np.argsort(vector.toarray()[0])[::-1]
        
        top_keywords = []
        for idx in sorted_coef_indexes:
            if vector[0, idx] > 0:
                top_keywords.append(feature_names[idx])
            if len(top_keywords) >= top_n:
                break
        return top_keywords
    except Exception:
        return []

def save_to_history(title, domain, is_fake, confidence):
    """ Lưu kết quả phân tích vào SQLite database """
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        label_str = "Tin Giả (Fake News)" if is_fake else "Tin Thật (True News)"
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO history (title, domain, result_label, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, domain, label_str, confidence, current_time))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Lỗi lưu DB: {e}")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    result_data = None
    preview = None
    meta_info = None
    keywords = []
    
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        url_input = request.form.get('url_input', '').strip()
        text_input = request.form.get('text_input', '').strip()
        
        content = ""
        title_val = "Văn bản tự nhập trực tiếp"
        domain_val = "Nội dung thủ công"

        if input_type == 'url':
            if url_input:
                article_data = extract_article_info(url_input)
                if article_data and len(article_data['text'].strip()) >= 50:
                    content = article_data['text']
                    title_val = article_data['title']
                    domain_val = article_data['domain']
                    
                    is_trusted = domain_val.lower() in TRUSTED_DOMAINS
                    domain_status = "Báo chính thống / Nguồn uy tín" if is_trusted else "Nguồn tin tự do / Cần kiểm chứng"
                    
                    meta_info = {
                        'title': title_val,
                        'authors': article_data['authors'],
                        'publish_date': article_data['publish_date'],
                        'domain': domain_val,
                        'domain_status': domain_status
                    }
                else:
                    result = "❌ Không thể cào dữ liệu từ URL này. Trang web có thể chặn bot hoặc liên kết không hợp lệ."
        else:
            content = text_input
            if len(content.strip()) > 20:
                title_val = content[:40] + "..."

        if content and len(content.strip()) >= 10:
            preview = content[:350] + ("..." if len(content) > 350 else "")
            
            # Xử lý làm sạch văn bản đa ngôn ngữ
            cleaned = clean_multilingual_text(content)
            keywords = get_top_keywords(cleaned, top_n=6)
            
            # Dự đoán bằng mô hình AI
            vectorized = tfidf.transform([cleaned])
            pred = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0]
            
            confidence = round(prob[1] * 100, 2) if pred == 1 else round(prob[0] * 100, 2)
            is_fake_bool = True if pred == 1 else False
            
            result_data = {
                'is_fake': is_fake_bool,
                'prob': confidence
            }
            
            # Lưu tự động vào lịch sử cơ sở dữ liệu
            save_to_history(title_val, domain_val, is_fake_bool, confidence)
                
        elif not result:
            result = "⚠️ Vui lòng cung cấp nội dung bài viết hợp lệ (ít nhất 10 ký tự)."

    return render_template('index.html', result=result, result_data=result_data, preview=preview, meta=meta_info, keywords=keywords)

@app.route('/history')
def history():
    """ Route hiển thị trang danh sách lịch sử kiểm tra """
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title, domain, result_label, confidence, created_at FROM history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', history_rows=rows)

if __name__ == '__main__':
    app.run(debug=True, port=5000)