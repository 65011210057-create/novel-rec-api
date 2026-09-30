from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Novel Recommendation Cloud API")

# โหลดโมเดลตอนเริ่ม Server
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class BookItem(BaseModel):
    book_id: int
    text: str

class CalculationPayload(BaseModel):
    books: List[BookItem]

@app.get("/")
def root():
    return {"status": "API is online"}

@app.post("/calculate-similarity")
def calculate_similarity(payload: CalculationPayload):
    books = payload.books
    if len(books) <= 1:
        return {"recommendations": []}

    book_ids = [b.book_id for b in books]
    texts = [b.text for b in books]

    vectors = model.encode(texts, show_progress_bar=False)
    similarity_matrix = cosine_similarity(vectors)

    results = []
    for i, book_id in enumerate(book_ids):
        scores = similarity_matrix[i]
        ranked = []
        for j, score in enumerate(scores):
            if book_ids[j] != book_id:
                ranked.append({"recommend_book_id": int(book_ids[j]), "similarity": round(float(score), 4)})
        
        ranked.sort(key=lambda x: x["similarity"], reverse=True)
        for item in ranked[:5]:
            results.append({
                "book_id": int(book_id),
                "recommend_book_id": item["recommend_book_id"],
                "similarity": item["similarity"]
            })

    return {"recommendations": results}