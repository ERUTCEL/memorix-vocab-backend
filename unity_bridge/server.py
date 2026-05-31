"""
FastAPI server — PostgreSQL/SQLite 기반, 다중 유저 지원.
uvicorn unity_bridge.server:app --host 0.0.0.0 --port $PORT
"""
import json, math, os, re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import engine, get_db
from db.models import Base, OxfordWord, UserWord, UserWordFSRS, WordStat
from db import crud
from unity_bridge.algorithms import (
    get_k_factor, update_user_rating, sigmoid_irt,
    init_fsrs_card, process_review, retrievability,
    estimate_user_rating, build_daily_schedule,
    cat_select_next, cat_map_theta, cat_se_theta,
    CAT_MAX_QUESTIONS, CAT_MIN_QUESTIONS, CAT_SE_THRESHOLD,
)

ROOT = Path(__file__).parent.parent

BUCKETS = {
    "A1": (50, 200, 100),
    "A2": (200, 350, 250),
    "B1": (350, 500, 400),
    "B2": (500, 650, 550),
    "C1": (650, 800, 700),
}
QUESTIONS_PER_BUCKET = 20


# ── Startup ──────────────────────────────────────────────────────────────────

def startup() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        _seed_oxford_if_empty()
    except Exception as e:
        print(f"[STARTUP] DB 초기화 실패 (나중에 재시도 가능): {e}")


def _seed_oxford_if_empty() -> None:
    from db.database import SessionLocal
    db = SessionLocal()
    try:
        if crud.oxford_count(db) > 0:
            return
        refined_path = ROOT / "models/refined_db.json"
        if not refined_path.exists():
            print("[SEED] models/refined_db.json 없음. Oxford DB 시딩 건너뜀.")
            return
        with open(refined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = crud.seed_oxford_from_json(db, data["words"])
        print(f"[SEED] Oxford DB 시딩 완료: {count}단어")
    finally:
        db.close()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Memorix Vocab Backend", version="2.0.0", on_startup=[startup])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def require_user(db: Session, user_id: str):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(404, detail="user_id 없음. POST /api/users/create 로 생성하세요.")
    return user


def require_onboarded(db: Session, user_id: str):
    user = require_user(db, user_id)
    if not user.onboarding_completed:
        raise HTTPException(400, detail="온보딩 미완료. /api/onboarding/cat/start 를 먼저 실행하세요.")
    return user


def _oxford_list(db: Session) -> list:
    """Return all oxford words as dicts for algorithm functions."""
    return [
        {"word": w.word, "rating_refined": w.rating_refined, "pos": w.pos, "meaning": w.meaning}
        for w in crud.get_all_oxford(db)
    ]


def _parse_json_robust(raw: str) -> dict | None:
    def _clean(s):
        s = re.sub(r"//[^\n]*", "", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r",\s*([\}\]])", r"\1", s)
        return s.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_clean(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(_clean(m.group()))
        except json.JSONDecodeError:
            pass
    return None


def _claude_client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, detail="ANTHROPIC_API_KEY 환경변수 없음")
    return anthropic.Anthropic(api_key=key)


def _apply_fsrs_update(fsrs_rec: UserWordFSRS, rating_given: int, today: date) -> None:
    """Update a UserWordFSRS ORM object in-place using FSRS algorithm."""
    if fsrs_rec.state == "queued" or fsrs_rec.review_count == 0:
        data = init_fsrs_card(rating_given, today)
        fsrs_rec.stability = data["stability"]
        fsrs_rec.difficulty = data["difficulty"]
        due_str = data["due_date"]
        fsrs_rec.due_date = date.fromisoformat(due_str[:10]) if due_str else None
        fsrs_rec.state = data["state"]
        fsrs_rec.review_count = 1
        fsrs_rec.last_review = today
        fsrs_rec.first_exposure = True
        fsrs_rec.review_history = [[0, rating_given]]

    elif fsrs_rec.state == "learning":
        elapsed = (today - fsrs_rec.last_review).days if fsrs_rec.last_review else 0
        prev_count = fsrs_rec.review_count
        data = init_fsrs_card(rating_given, today)
        fsrs_rec.stability = data["stability"]
        fsrs_rec.difficulty = data["difficulty"]
        due_str = data["due_date"]
        fsrs_rec.due_date = date.fromisoformat(due_str[:10]) if due_str else None
        fsrs_rec.state = data["state"]
        fsrs_rec.review_count = prev_count + 1
        fsrs_rec.last_review = today
        fsrs_rec.first_exposure = False
        history = list(fsrs_rec.review_history or [])
        history.append([elapsed, rating_given])
        fsrs_rec.review_history = history

    else:
        elapsed = (today - fsrs_rec.last_review).days if fsrs_rec.last_review else 0
        current = crud.fsrs_to_dict(fsrs_rec)
        current["last_review"] = current["last_review"] or today.isoformat()
        data = process_review(current, rating_given, today)
        fsrs_rec.stability = data["stability"]
        fsrs_rec.difficulty = data["difficulty"]
        due_str = data["due_date"]
        fsrs_rec.due_date = date.fromisoformat(due_str[:10]) if due_str else None
        fsrs_rec.state = data["state"]
        fsrs_rec.review_count = data["review_count"]
        fsrs_rec.last_review = today
        fsrs_rec.first_exposure = False
        history = list(fsrs_rec.review_history or [])
        history.append([elapsed, rating_given])
        fsrs_rec.review_history = history


# ── Request models ────────────────────────────────────────────────────────────

class QuizAnswer(BaseModel):
    order: int
    word: str
    correct: bool
    response_time_ms: Optional[int] = None


class QuizSubmit(BaseModel):
    user_id: str
    answers: list[QuizAnswer]


class CatStart(BaseModel):
    user_id: str


class CatAnswerBody(BaseModel):
    user_id: str
    word: str
    correct: bool
    response_time_ms: Optional[int] = None


class SessionAnswer(BaseModel):
    word: str
    correct: Optional[bool] = None
    rating_given: int = 3  # 1=Again 2=Hard 3=Good 4=Easy
    response_time_ms: Optional[int] = None


class SessionResult(BaseModel):
    user_id: str
    answers: list[SessionAnswer]


class SessionCheckpoint(BaseModel):
    user_id: str
    answers: list[SessionAnswer]
    remaining_words: list[str]


class AiRecommendRequest(BaseModel):
    user_id: str
    domain: Optional[str] = None
    count: int = 10


class ScheduleRecommendRequest(BaseModel):
    user_id: str
    daily_minutes: int = 30


class InitDefaultRequest(BaseModel):
    user_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    try:
        oxford_count = crud.oxford_count(db)
        db_ok = True
    except Exception:
        oxford_count = 0
        db_ok = False
    return {
        "status": "ok" if db_ok else "db_error",
        "timestamp": datetime.now().isoformat(),
        "oxford_words": oxford_count,
        "db_ok": db_ok,
    }


# ── User management ────────────────────────────────────────────────────────────

@app.post("/api/users/create")
def create_user(db: Session = Depends(get_db)):
    """새 유저 생성. user_id를 로컬(PlayerPrefs)에 저장해 이후 모든 요청에 사용."""
    user = crud.create_user(db)
    return {"user_id": user.user_id, "status": "created"}


@app.get("/api/user/profile")
def get_user_profile(user_id: str = Query(...), db: Session = Depends(get_db)):
    user = require_user(db, user_id)
    return crud.user_to_dict(user)


@app.post("/api/init-default")
def init_default(body: InitDefaultRequest, db: Session = Depends(get_db)):
    """CSV 없이 빈 단어 풀로 시작. 스케줄러가 Oxford DB 보충으로 단어 제공."""
    user = require_user(db, body.user_id)
    oxford_count = crud.oxford_count(db)
    return {
        "status": "ok",
        "user_id": user.user_id,
        "total_words": 0,
        "oxford_available": oxford_count,
        "message": "Oxford DB에서 자동 보충합니다. /api/onboarding/cat/start 를 실행하세요.",
    }


# ── CSV Upload ────────────────────────────────────────────────────────────────

@app.post("/api/upload-csv")
async def upload_csv(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """유저 CSV 업로드 → Oxford 매칭 + 미등록 단어 Claude API 예측 → 단어 풀 등록."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="CSV 파일만 허용됩니다.")

    user = require_user(db, user_id)
    content = (await file.read()).decode("utf-8-sig", errors="replace")

    import io, csv as _csv
    reader = _csv.DictReader(io.StringIO(content))
    headers = [h.strip().upper() for h in (reader.fieldnames or [])]

    word_col = next((h for h in headers if h in ("WORD", "단어")), None)
    if word_col is None:
        raise HTTPException(400, detail=f"WORD 또는 단어 컬럼이 없습니다. 실제 컬럼: {headers}")

    orig_headers = reader.fieldnames or []
    header_map = {h.strip().upper(): h for h in orig_headers}
    real_word_col = header_map[word_col]

    # parse meaning column if present
    meaning_col_upper = next((h for h in headers if h in ("MEANING", "뜻")), None)
    real_meaning_col = header_map.get(meaning_col_upper) if meaning_col_upper else None

    seen, user_words_clean = set(), []
    for row in reader:
        word = str(row.get(real_word_col, "") or "").lower().strip()
        word = re.sub(r"[^\w]", "", word)
        if word and word not in seen:
            seen.add(word)
            meaning = str(row.get(real_meaning_col, "") or "").strip() if real_meaning_col else None
            user_words_clean.append({"word": word, "meaning": meaning or None})

    if not user_words_clean:
        raise HTTPException(400, detail="CSV가 비어있습니다.")

    oxford_map = {w.word: w for w in crud.get_all_oxford(db)}
    matched, unmatched = [], []
    for entry in user_words_clean:
        word = entry["word"]
        if word in oxford_map:
            matched.append(word)
            crud.upsert_fsrs_queued(db, user_id, word, "oxford")
        else:
            unmatched.append(entry)

    # Predict ratings for unmatched words via embeddings or Claude API
    predicted_count = 0
    api_verified_count = 0

    if unmatched:
        # Try KNN first (embeddings_cache.pkl)
        knn_results = _knn_predict_words([u["word"] for u in unmatched])
        if knn_results:
            for r in knn_results:
                source = "predicted"
                rating = r["predicted_rating"]
                confidence = r["confidence"]
                if r.get("low_confidence"):
                    source = "predicted_low_conf"
                crud.upsert_user_word(db, r["word"], rating, confidence, source)
                crud.upsert_fsrs_queued(db, user_id, r["word"], "user")
                predicted_count += 1
        else:
            # Fallback: Claude API batch prediction
            api_result = _claude_rate_words(
                [u["word"] for u in unmatched],
                {u["word"]: u["meaning"] for u in unmatched if u["meaning"]},
            )
            for entry in unmatched:
                word = entry["word"]
                info = api_result.get(word, {})
                rating = info.get("rating", 400)
                source = "api_verified" if word in api_result else "predicted"
                confidence = 0.9 if word in api_result else 0.6
                crud.upsert_user_word(db, word, rating, confidence, source)
                crud.upsert_fsrs_queued(db, user_id, word, "user")
                if word in api_result:
                    api_verified_count += 1
                else:
                    predicted_count += 1

    db.commit()

    return {
        "status": "ok",
        "user_id": user_id,
        "total_words": len(matched) + len(unmatched),
        "oxford_matched": len(matched),
        "predicted": predicted_count,
        "api_verified": api_verified_count,
    }


def _knn_predict_words(words: list[str]) -> list[dict] | None:
    """Try KNN via embeddings_cache.pkl. Returns None if unavailable."""
    try:
        import pickle, numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        pkl_path = ROOT / "models/embeddings_cache.pkl"
        if not pkl_path.exists():
            return None

        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)

        st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        embeddings = st_model.encode(words, batch_size=64, show_progress_bar=False)
        db_embeddings = cache["embeddings"]
        db_ratings = np.array(cache["ratings"])

        results = []
        for word, emb in zip(words, embeddings):
            sims = cosine_similarity([emb], db_embeddings)[0]
            top_idx = np.argsort(sims)[-5:][::-1]
            top_sims, top_ratings = sims[top_idx], db_ratings[top_idx]
            w_sum = top_sims.sum()
            predicted = float(np.dot(top_sims, top_ratings) / w_sum) if w_sum else float(np.mean(top_ratings))
            std = float(np.std(top_ratings))
            confidence = max(0.5, 1.0 - std / 300.0)
            results.append({
                "word": word, "predicted_rating": round(predicted),
                "confidence": round(confidence, 3), "low_confidence": confidence < 0.7 or std > 150,
            })
        return results
    except Exception:
        return None


def _claude_rate_words(words: list[str], meaning_hints: dict = None) -> dict:
    """Batch-rate words via Claude API. Returns {word: {rating, cefr}}."""
    if not words:
        return {}
    try:
        client = _claude_client()
        prompt = (
            "다음 영어 단어들의 CEFR 레벨과 레이팅을 JSON으로 반환하세요.\n"
            "레이팅 기준: A1=100, A2=250, B1=400, B2=550, C1=700\n\n"
            f"단어 목록: {json.dumps(words, ensure_ascii=False)}\n"
        )
        if meaning_hints:
            prompt += f"의미 힌트(참고용): {json.dumps(meaning_hints, ensure_ascii=False)}\n"
        prompt += '\n응답 형식 (JSON만):\n{"word": {"rating": 550, "cefr": "B2"}, ...}'

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system="영어 단어 난이도 평가 전문가. JSON만 반환.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = _parse_json_robust(raw.strip())
        return result or {}
    except Exception:
        return {}


# ── Onboarding ────────────────────────────────────────────────────────────────

@app.get("/api/onboarding/quiz")
def get_quiz(user_id: str = Query(...), db: Session = Depends(get_db)):
    """퀴즈 단어 샘플링 반환. Oxford DB에서 CEFR 구간별 균등 샘플링."""
    import random
    require_user(db, user_id)
    oxford_words = crud.get_all_oxford(db)
    if len(oxford_words) < 10:
        raise HTTPException(500, detail="Oxford DB가 비어있습니다.")

    questions, order = [], 1
    for bucket_name, (lo, hi, center) in BUCKETS.items():
        pool = [w for w in oxford_words if lo <= w.rating_refined < hi]
        if len(pool) < QUESTIONS_PER_BUCKET:
            margin = 50
            while len(pool) < QUESTIONS_PER_BUCKET and margin <= 200:
                pool = [w for w in oxford_words if (lo - margin) <= w.rating_refined < (hi + margin)]
                margin += 50
        sampled = random.sample(pool, min(QUESTIONS_PER_BUCKET, len(pool)))
        for w in sampled:
            questions.append({
                "order": order, "word": w.word, "rating": w.rating_refined,
                "bucket": bucket_name, "correct": None, "response_time_ms": None,
            })
            order += 1

    random.shuffle(questions)
    for i, q in enumerate(questions, 1):
        q["order"] = i

    return {"total_questions": len(questions), "questions": questions}


@app.post("/api/onboarding/submit")
def submit_quiz(body: QuizSubmit, db: Session = Depends(get_db)):
    """퀴즈 결과 처리 → userRating 초기화."""
    user = require_user(db, body.user_id)
    if not body.answers:
        raise HTTPException(400, detail="answers가 비어있습니다.")

    bucket_correct = {b: 0 for b in BUCKETS}
    bucket_total = {b: 0 for b in BUCKETS}

    oxford_map = {w.word: w for w in crud.get_all_oxford(db)}
    for ans in body.answers:
        word_obj = oxford_map.get(ans.word)
        if not word_obj:
            continue
        r = word_obj.rating_refined
        for bname, (lo, hi, _) in BUCKETS.items():
            if lo <= r < hi:
                bucket_total[bname] += 1
                if ans.correct:
                    bucket_correct[bname] += 1
                break

    bucket_accuracy = {
        b: bucket_correct[b] / bucket_total[b] if bucket_total[b] > 0 else 0.0
        for b in BUCKETS
    }
    centers = [BUCKETS[b][2] for b in BUCKETS]
    accuracies = [bucket_accuracy[b] for b in BUCKETS]
    user_rating = estimate_user_rating(centers, accuracies)

    user.user_rating = user_rating
    user.rating_history = [user_rating]
    user.k_factor = get_k_factor(0)
    user.onboarding_completed = True
    crud.update_user(db, user)

    result = crud.user_to_dict(user)
    result["onboarding_accuracy"] = {b: round(acc, 3) for b, acc in bucket_accuracy.items()}
    return result


@app.post("/api/onboarding/cat/start")
def cat_start(body: CatStart, db: Session = Depends(get_db)):
    """CAT 온보딩 시작 → 첫 문항 반환."""
    user = require_user(db, body.user_id)

    oxford_words = crud.get_all_oxford(db)
    if len(oxford_words) < 10:
        raise HTTPException(500, detail="Oxford DB가 비어있습니다.")

    words = [{"word": w.word, "rating": w.rating_refined} for w in oxford_words]
    theta = 400.0
    from unity_bridge.algorithms import cat_select_next
    first = cat_select_next(theta, words, set())
    if not first:
        raise HTTPException(500, detail="단어 풀이 비어있습니다.")

    cat_state = {
        "theta": theta, "se": 999.0, "responses": [],
        "asked_words": [first["word"]], "question_num": 1, "done": False,
    }
    user.cat_state = cat_state
    crud.update_user(db, user)

    return {
        "done": False, "question_num": 1, "max_questions": CAT_MAX_QUESTIONS,
        "theta": round(theta), "word": first["word"], "rating": first["rating"],
    }


@app.post("/api/onboarding/cat/answer")
def cat_answer(body: CatAnswerBody, db: Session = Depends(get_db)):
    """CAT 한 문항 제출 → 다음 문항 or 완료 반환."""
    user = require_user(db, body.user_id)
    state = user.cat_state
    if not state or state.get("done"):
        raise HTTPException(400, detail="CAT가 시작되지 않았습니다. /api/onboarding/cat/start 먼저 실행.")

    oxford_map = {w.word: w.rating_refined for w in crud.get_all_oxford(db)}
    rating = oxford_map.get(body.word, round(state["theta"]))

    state["responses"].append({"word": body.word, "rating": rating, "correct": body.correct})
    new_theta = cat_map_theta(state["responses"])
    new_se = cat_se_theta(new_theta, state["responses"])
    state["theta"] = new_theta
    state["se"] = new_se
    q_num = state["question_num"]

    if q_num >= CAT_MAX_QUESTIONS or (q_num >= CAT_MIN_QUESTIONS and new_se < CAT_SE_THRESHOLD):
        # Finalize
        user_rating = round(new_theta)
        user.user_rating = user_rating
        user.rating_history = [user_rating]
        user.k_factor = get_k_factor(0)
        user.onboarding_completed = True
        state["done"] = True
        user.cat_state = state
        crud.update_user(db, user)
        return {
            "done": True, "question_num": q_num,
            "user_profile": crud.user_to_dict(user),
        }

    state["question_num"] += 1
    words = [{"word": w.word, "rating": w.rating_refined} for w in crud.get_all_oxford(db)]
    next_item = cat_select_next(new_theta, words, set(state["asked_words"]))
    if not next_item:
        user_rating = round(new_theta)
        user.user_rating = user_rating
        user.rating_history = [user_rating]
        user.k_factor = get_k_factor(0)
        user.onboarding_completed = True
        state["done"] = True
        user.cat_state = state
        crud.update_user(db, user)
        return {"done": True, "question_num": q_num, "user_profile": crud.user_to_dict(user)}

    state["asked_words"].append(next_item["word"])
    user.cat_state = state
    crud.update_user(db, user)

    return {
        "done": False, "question_num": state["question_num"],
        "max_questions": CAT_MAX_QUESTIONS, "theta": round(new_theta),
        "word": next_item["word"], "rating": next_item["rating"],
    }


# ── Schedule ──────────────────────────────────────────────────────────────────

@app.get("/api/schedule/today")
def get_today_schedule(
    user_id: str = Query(...),
    daily_limit: int = Query(default=100, ge=1, le=1000),
    total_words: Optional[int] = Query(default=None, ge=1),
    days: Optional[int] = Query(default=None, ge=1),
    db: Session = Depends(get_db),
):
    user = require_onboarded(db, user_id)

    # study_plan이 있으면 daily_limit 자동 적용
    plan = user.study_plan
    plan_applied = False
    if plan and plan.get("confirmed"):
        start = date.fromisoformat(plan["start_date"])
        idx = (date.today() - start).days
        daily_plan = plan["daily_plan"]
        daily_limit = daily_plan[min(idx, len(daily_plan) - 1)]
        plan_applied = True

    if not plan_applied and total_words is not None and days is not None:
        daily_limit = round((total_words / days) / (1 - 0.4))

    rated_words = crud.build_rated_words_dict(db, user_id)
    user_profile = crud.user_to_dict(user)
    oxford_words = _oxford_list(db)

    schedule = build_daily_schedule(rated_words, user_profile, daily_limit, date.today(), oxford_words)
    return schedule


@app.post("/api/schedule/recommend")
def schedule_recommend(body: ScheduleRecommendRequest, db: Session = Depends(get_db)):
    """Claude API로 학습 플랜 추천 → study_plan 저장."""
    user = require_onboarded(db, body.user_id)
    client = _claude_client()

    rated = crud.build_rated_words_dict(db, body.user_id)
    buckets = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0}
    for w in rated["words"]:
        r = w.get("rating", 0)
        if r < 175:
            buckets["A1"] += 1
        elif r < 325:
            buckets["A2"] += 1
        elif r < 475:
            buckets["B1"] += 1
        elif r < 625:
            buckets["B2"] += 1
        else:
            buckets["C1"] += 1

    total_words_count = len(rated["words"])
    daily_capacity = body.daily_minutes * 2
    cefr = (
        "A1" if user.user_rating < 175 else
        "A2" if user.user_rating < 325 else
        "B1" if user.user_rating < 475 else
        "B2" if user.user_rating < 625 else "C1"
    )
    user_ctx = {
        "total_words": total_words_count,
        "daily_minutes": body.daily_minutes,
        "daily_capacity": daily_capacity,
        "user_rating": user.user_rating,
        "total_sessions": user.total_sessions,
        "cefr_level": cefr,
        "difficulty_distribution": buckets,
    }

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="당신은 영어 학습 플랜 전문가입니다. 반드시 JSON만 반환하세요.",
        messages=[{
            "role": "user",
            "content": (
                f"다음 유저 데이터를 분석해서 최적 학습 플랜을 추천하세요.\n"
                f"유저 데이터: {json.dumps(user_ctx, ensure_ascii=False)}\n\n"
                "응답 형식 (JSON만, 설명 없이):\n"
                '{"recommended_days": 10, "recommended_daily_limit": 60, '
                '"daily_plan": [70, 70, 65, 60, 60, 55, 55, 50, 50, 50], '
                '"reason": "추천 이유 한 줄"}'
            ),
        }],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    plan_data = _parse_json_robust(raw.strip())
    if not plan_data:
        raise HTTPException(500, detail="Claude 응답 파싱 실패")

    user.study_plan = {
        "start_date": date.today().isoformat(),
        "daily_plan": plan_data["daily_plan"],
        "confirmed": True,
    }
    crud.update_user(db, user)

    return {
        "recommended_days": plan_data["recommended_days"],
        "recommended_daily_limit": plan_data["recommended_daily_limit"],
        "daily_plan": plan_data["daily_plan"],
        "reason": plan_data.get("reason", ""),
        "total_words": total_words_count,
        "user_rating": user.user_rating,
    }


# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/api/session/result")
def submit_session_result(body: SessionResult, db: Session = Depends(get_db)):
    """세션 결과 제출 → FSRS + userRating 업데이트 + word_stats INSERT."""
    user = require_user(db, body.user_id)
    if not body.answers:
        raise HTTPException(400, detail="answers가 비어있습니다.")

    today = date.today()
    oxford_map = {w.word: w for w in crud.get_all_oxford(db)}
    user_word_map = {w.word: w for w in db.query(UserWord).all()}

    k = get_k_factor(user.total_sessions)
    checkpoint_answered = set(user.checkpoint_answered or [])
    correct_count, total_tracked = 0, 0

    for ans in body.answers:
        word = ans.word.lower().strip()
        rating_given = ans.rating_given
        correct = ans.correct
        if correct is None:
            correct = rating_given > 1

        # Get or auto-create FSRS record for supplement words
        fsrs_rec = crud.get_fsrs(db, body.user_id, word)
        if fsrs_rec is None:
            if word in oxford_map:
                fsrs_rec = crud.upsert_fsrs_queued(db, body.user_id, word, "oxford")
            elif word in user_word_map:
                fsrs_rec = crud.upsert_fsrs_queued(db, body.user_id, word, "user")
            else:
                continue

        # Word rating for IRT
        if fsrs_rec.word_source == "oxford" and word in oxford_map:
            word_rating = oxford_map[word].rating_refined
        elif word in user_word_map:
            word_rating = user_word_map[word].rating_predicted
        else:
            word_rating = user.user_rating

        _apply_fsrs_update(fsrs_rec, rating_given, today)

        if word not in checkpoint_answered:
            user.user_rating = update_user_rating(user.user_rating, word_rating, correct, k)

        if correct is not None:
            total_tracked += 1
            if correct:
                correct_count += 1

        crud.insert_word_stat(db, body.user_id, word, correct, rating_given, ans.response_time_ms)

    user.total_sessions += 1
    user.k_factor = get_k_factor(user.total_sessions)
    if total_tracked > 0:
        user.last_session_accuracy = round(correct_count / total_tracked, 3)
    user.checkpoint_answered = []
    history = list(user.rating_history or [])
    history.append(user.user_rating)
    user.rating_history = history

    db.commit()
    return crud.user_to_dict(user)


@app.post("/api/session/checkpoint")
def session_checkpoint(body: SessionCheckpoint, db: Session = Depends(get_db)):
    """세션 중간 정답률 → 남은 단어 재정렬 + user_rating 실시간 업데이트."""
    user = require_user(db, body.user_id)
    if not body.answers:
        raise HTTPException(400, detail="answers가 비어있습니다.")

    accuracy = sum(1 for a in body.answers if a.correct) / len(body.answers)

    oxford_map = {w.word: w.rating_refined for w in crud.get_all_oxford(db)}
    rated = crud.build_rated_words_dict(db, body.user_id)
    word_rating = {w["word"]: w["rating"] for w in rated["words"]}
    for w, r in oxford_map.items():
        if w not in word_rating:
            word_rating[w] = r

    k = get_k_factor(user.total_sessions)
    newly_answered = []
    for a in body.answers:
        if a.correct is not None:
            wr = word_rating.get(a.word, user.user_rating)
            user.user_rating = update_user_rating(user.user_rating, wr, a.correct, k)
            newly_answered.append(a.word)

    existing = set(user.checkpoint_answered or [])
    user.checkpoint_answered = list(existing | set(newly_answered))
    db.commit()

    remaining = list(body.remaining_words)
    user_rating = user.user_rating

    if accuracy > 0.75:
        hard = [w for w in remaining if word_rating.get(w, user_rating) >= user_rating + 100]
        rest = [w for w in remaining if word_rating.get(w, user_rating) < user_rating + 100]
        remaining = hard + rest
        mode = "hard"
    elif accuracy < 0.50:
        easy = [w for w in remaining if word_rating.get(w, user_rating) <= user_rating - 100]
        rest = [w for w in remaining if word_rating.get(w, user_rating) > user_rating - 100]
        today_str = date.today().isoformat()
        answered_set = {a.word for a in body.answers}
        extra = [
            w["word"] for w in rated["words"]
            if w.get("learned") and w.get("fsrs") and w["fsrs"].get("due_date")
            and w["fsrs"]["due_date"][:10] <= today_str
            and w["word"] not in set(body.remaining_words)
            and w["word"] not in answered_set
        ][:10]
        remaining = easy + rest + extra
        mode = "easy"
    else:
        mode = "normal"

    return {
        "accuracy": round(accuracy, 3),
        "user_rating": user_rating,
        "updated_user_rating": user_rating,
        "mode": mode,
        "remaining_words": remaining,
    }


# ── Words ─────────────────────────────────────────────────────────────────────

@app.get("/api/words/all")
def get_all_words(user_id: str = Query(...), db: Session = Depends(get_db)):
    require_user(db, user_id)
    return crud.build_rated_words_dict(db, user_id)


@app.get("/api/recommend/words")
def recommend_words(
    user_id: str = Query(...),
    count: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Oxford DB에서 유저 수준 근접 미학습 단어 추천."""
    user = require_onboarded(db, user_id)
    user_rating = user.user_rating

    pool_words = {r.word for r in crud.get_all_fsrs(db, user_id)}
    all_oxford = crud.get_all_oxford(db)

    # weak zone: words the user answered Again (rating=1)
    from db.models import WordStat as WS
    weak_stats = db.query(WS.word).filter(WS.user_id == user_id, WS.rating_given == 1).all()
    weak_words = {r.word for r in weak_stats}
    weak_ratings = [
        w.rating_refined for w in all_oxford if w.word in weak_words
    ]
    weak_center = int(sum(weak_ratings) / len(weak_ratings)) if weak_ratings else user_rating

    candidates = [
        w for w in all_oxford
        if w.word not in pool_words and abs(w.rating_refined - user_rating) <= 150
    ]

    def _score(w):
        proximity = abs(w.rating_refined - user_rating)
        weak_boost = max(0, 100 - abs(w.rating_refined - weak_center))
        return proximity - weak_boost

    candidates.sort(key=_score)
    return {
        "user_rating": user_rating,
        "weak_center": weak_center,
        "count": min(count, len(candidates)),
        "words": [
            {"word": w.word, "rating": w.rating_refined, "pos": w.pos}
            for w in candidates[:count]
        ],
    }


@app.post("/api/recommend/ai")
def recommend_ai(body: AiRecommendRequest, db: Session = Depends(get_db)):
    """Claude API로 유저 맞춤 단어 추천 → user_words + user_word_fsrs 자동 편입."""
    user = require_onboarded(db, body.user_id)
    client = _claude_client()

    rated = crud.build_rated_words_dict(db, body.user_id)
    all_user_words = {w["word"] for w in rated["words"]}
    learned_words = [w["word"] for w in rated["words"] if w.get("learned")]

    weak_stats = db.query(WordStat.word).filter(
        WordStat.user_id == body.user_id, WordStat.rating_given == 1
    ).all()
    weak_words = {r.word for r in weak_stats}
    weak_ratings = [
        r.rating_refined for r in crud.get_all_oxford(db) if r.word in weak_words
    ]
    weak_range = None
    if weak_ratings:
        wc = int(sum(weak_ratings) / len(weak_ratings))
        weak_range = [wc - 100, wc + 100]

    cefr = (
        "A1" if user.user_rating < 175 else "A2" if user.user_rating < 325 else
        "B1" if user.user_rating < 475 else "B2" if user.user_rating < 625 else "C1"
    )
    user_ctx = {
        "user_rating": user.user_rating,
        "cefr_level": cefr,
        "domain": body.domain,
        "weak_rating_range": weak_range,
        "recently_learned": learned_words[-20:],
        "count": body.count,
    }

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="영어 단어 학습 추천 전문가. JSON만 반환.",
        messages=[{
            "role": "user",
            "content": (
                f"다음 유저 프로필을 분석해서 영어 단어 {body.count}개를 추천하세요.\n"
                f"유저 프로필: {json.dumps(user_ctx, ensure_ascii=False)}\n\n"
                '응답 형식 (JSON만):\n'
                '{"recommended": [{"word": "...", "reason": "...", "cefr": "B2", "rating": 550}]}'
            ),
        }],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = _parse_json_robust(raw.strip())
    if not result:
        raise HTTPException(500, detail="Claude 응답 파싱 실패")

    added = []
    for item in result.get("recommended", []):
        word = (item.get("word") or "").lower().strip()
        if not word or word in all_user_words:
            continue
        rating = item.get("rating", user.user_rating)
        crud.upsert_user_word(db, word, rating, 0.8, "ai_recommended")
        crud.upsert_fsrs_queued(db, body.user_id, word, "user")
        all_user_words.add(word)
        added.append({"word": word, "rating": rating,
                      "reason": item.get("reason"), "cefr": item.get("cefr")})

    db.commit()
    return {"added_count": len(added), "words": added}


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/api/admin/update-ratings")
def update_ratings(db: Session = Depends(get_db)):
    """word_stats 누적 정답률로 oxford_words.rating_refined 자동 보정 (50회 이상)."""
    stats = crud.get_word_stats_aggregated(db, min_count=50)
    if not stats:
        return {"updated": 0, "message": "50회 이상 데이터 없음"}

    updated = 0
    for s in stats:
        ox = crud.get_oxford_by_word(db, s["word"])
        if not ox:
            continue
        accuracy = s["accuracy"]
        # target accuracy for this word's difficulty vs. user pool
        # simple heuristic: accuracy < 0.5 → rating too easy (lower), > 0.8 → too hard (raise)
        delta = 0
        if accuracy < 0.40:
            delta = -20  # word is harder than rated → lower rating (easier for user)
        elif accuracy < 0.50:
            delta = -10
        elif accuracy > 0.85:
            delta = +15  # word is easier than rated → raise rating
        elif accuracy > 0.75:
            delta = +8

        if delta != 0:
            new_rating = max(ox.rating_base - 100, min(ox.rating_base + 100, ox.rating_refined + delta))
            ox.rating_refined = new_rating
            updated += 1

    db.commit()
    return {"updated": updated, "total_analyzed": len(stats)}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
