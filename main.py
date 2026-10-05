from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pymysql
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="Novel Recommendation TF-IDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลเชื่อมต่อ TiDB Cloud
DB_HOST = os.getenv("DB_HOST", "gateway01.ap-southeast-1.prod.aws.tidbcloud.com")
DB_USER = os.getenv("DB_USER", "4JodNqEkbc1nEbH.root")
DB_PASS = os.getenv("DB_PASS", "zF4DHIXiUrHylslj")
DB_NAME = os.getenv("DB_NAME", "thai_novel")
DB_PORT = int(os.getenv("DB_PORT", 4000))

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        charset="utf8mb4",
        ssl={"ssl": True}
    )

def run_tfidf_calculation(category_id: int):
    try:
        conn = get_db_connection()
        sql = "SELECT Book_id, Title, Blurb FROM book WHERE Category_id = %s ORDER BY Book_id"
        df = pd.read_sql(sql, conn, params=(category_id,))
        
        if len(df) <= 1:
            conn.close()
            return

        # รวมข้อความสำหรับทำ TF-IDF
        df["content"] = df["Title"].fillna("") + " " + df["Blurb"].fillna("")

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df["content"])
        sim_matrix = cosine_similarity(tfidf_matrix)
        book_ids = df["Book_id"].tolist()

        cursor = conn.cursor()
        # ล้างผลลัพธ์เดิมเฉพาะหมวดนี้
        cursor.execute("DELETE FROM recommendation_sentence_same_category WHERE book_id IN (SELECT Book_id FROM book WHERE Category_id = %s)", (category_id,))

        for i, b_id in enumerate(book_ids):
            scores = sim_matrix[i]
            ranked = []
            for j, score in enumerate(scores):
                if book_ids[j] != b_id:
                    ranked.append((int(book_ids[j]), float(score)))
            
            ranked.sort(key=lambda x: x[1], reverse=True)
            for rec_id, score in ranked[:5]:
                cursor.execute("""
                    INSERT INTO recommendation_sentence_same_category (book_id, recommend_book_id, similarity)
                    VALUES (%s, %s, %s)
                """, (int(b_id), int(rec_id), round(score, 4)))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"คำนวณและบันทึกหมวด {category_id} สำเร็จ")
    except Exception as e:
        print(f"Error calculating TF-IDF: {e}")

@app.get("/")
def root():
    return {"status": "online"}

@app.get("/calculate")
def trigger_calculate(category_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_tfidf_calculation, category_id)
    return {"status": "processing", "category_id": category_id}
