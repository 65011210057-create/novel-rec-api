from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pymysql
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = FastAPI(title="Novel Recommendation TF-IDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # รวมข้อความชื่อเรื่องและคำโปรย
        df["content"] = df["Title"].fillna("").astype(str) + " " + df["Blurb"].fillna("").astype(str)

        # ใช้ char_wb ngram (ขนาด 2-4 ตัวอักษร) เพื่อให้รองรับภาษาไทยโดยไม่ต้องพึ่ง PyThaiNLP ซึ่งกิน RAM ต่ำมาก
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(df["content"])
        sim_matrix = cosine_similarity(tfidf_matrix)
        book_ids = df["Book_id"].tolist()

        cursor = conn.cursor()

        # คำนวณและบันทึกคะแนนแนะนำให้หนังสือทุกเล่มในหมวดนี้
        for i, b_id in enumerate(book_ids):
            scores = sim_matrix[i]
            ranked = []
            for j, score in enumerate(scores):
                if book_ids[j] != b_id:
                    ranked.append((int(book_ids[j]), float(score)))
            
            # เรียงจากคะแนนมากไปหาน้อย
            ranked.sort(key=lambda x: x[1], reverse=True)
            top_recs = ranked[:5]

            # ลบเฉพาะข้อมูลแนะนำของเล่มนี้ออกก่อน เพื่อไม่ให้ข้อมูลของเล่มอื่นในตารางเดิมสูญหาย
            cursor.execute("DELETE FROM recommendation_sentence_same_category WHERE book_id = %s", (int(b_id),))

            # เพิ่มรายการแนะนำ Top 5 ของเล่มนี้ลงตาราง
            for rec_id, score in top_recs:
                cursor.execute("""
                    INSERT INTO recommendation_sentence_same_category (book_id, recommend_book_id, similarity)
                    VALUES (%s, %s, %s)
                """, (int(b_id), int(rec_id), round(score, 4)))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"คำนวณ TF-IDF สำหรับหมวด {category_id} และอัปเดตลงตาราง recommendation_sentence_same_category สำเร็จ")
    except Exception as e:
        print(f"Error calculating TF-IDF: {e}")

@app.get("/")
def root():
    return {"status": "online"}

@app.get("/calculate")
def trigger_calculate(category_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_tfidf_calculation, category_id)
    return {"status": "processing", "category_id": category_id}
