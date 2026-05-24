"""
FastAPI Unity 브릿지 서버
uvicorn unity_bridge.server:app --host 0.0.0.0 --port 8000 --reload
"""
import json, math, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 스크립트 루트를 이 파일 기준으로 결정 (unity_bridge/ 의 부모)
ROOT = Path(__file__).parent.parent

def _get_k(total_sessions: int) -> int:
    if total_sessions <= 5:
        return 100
    elif total_sessions <= 20:
        return 50
    return 20


def _irt_update(user_rating: int, word_rating: int, correct: bool, k: int) -> int:
    expected = 1.0 / (1.0 + math.exp(-(user_rating - word_rating) / 150.0))
    actual = 1 if correct else 0
    return round(user_rating + k * (actual - expected))


app = FastAPI(title="Vocab Rating System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 유틸 ────────────────────────────────────────────────────────────────────

def load_json(rel_path: str) -> dict:
    path = ROOT / rel_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{rel_path} 파일 없음")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(rel_path: str, data: dict) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_script(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """프로젝트 루트에서 scripts/ 스크립트 실행. timeout 포함 모든 예외를 HTTPException으로 변환."""
    cmd = [sys.executable] + list(args)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, detail=f"{args[0]} 타임아웃 ({timeout}s 초과)")
    except Exception as exc:
        raise HTTPException(500, detail=f"{args[0]} 실행 오류: {exc}")


# ── 요청 모델 ────────────────────────────────────────────────────────────────

class QuizAnswer(BaseModel):
    order: int
    word: str
    correct: bool
    response_time_ms: int | None = None


class QuizSubmit(BaseModel):
    answers: list[QuizAnswer]


class CatAnswer(BaseModel):
    word: str
    correct: bool
    response_time_ms: int | None = None


class SessionAnswer(BaseModel):
    word: str
    correct: bool | None = None
    rating_given: int = 3  # 1=Again, 2=Hard, 3=Good, 4=Easy


class SessionResult(BaseModel):
    answers: list[SessionAnswer]


class SessionCheckpoint(BaseModel):
    answers: list[SessionAnswer]       # 이번 세션에서 이미 답한 것들
    remaining_words: list[str]         # 아직 안 푼 단어 목록


class AiRecommendRequest(BaseModel):
    domain: str | None = None          # e.g. "business", "academic", "travel"
    count: int = 10


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "ai3_ready": (ROOT / "models/refined_db.json").exists(),
        "words_ready": (ROOT / "output/rated_words.json").exists(),
        "user_ready": (ROOT / "output/user_profile.json").exists(),
        "schedule_ready": (ROOT / "output/daily_schedule.json").exists(),
    }


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """유저 CSV 업로드 → AI2 실행"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="CSV 파일만 허용됩니다.")

    content = await file.read()
    csv_path = ROOT / "input/user_words.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(content)

    result = run_script("scripts/ai2_rate_csv.py", "--input", "input/user_words.csv", timeout=600)
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "AI2 실행 실패")

    rated = load_json("output/rated_words.json")
    return {
        "status": "ok",
        "total_words": rated["total_words"],
        "oxford_matched": rated["oxford_matched"],
        "predicted": rated["predicted"],
        "api_verified": rated.get("api_verified", 0),
    }


@app.get("/api/onboarding/quiz")
def get_quiz():
    """퀴즈 단어 생성 및 반환"""
    result = run_script("scripts/ai1_onboarding.py", "--generate-quiz")
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "퀴즈 생성 실패")
    return load_json("output/onboarding_quiz.json")


@app.post("/api/onboarding/submit")
def submit_quiz(body: QuizSubmit):
    """퀴즈 결과 제출 → userRating 초기화"""
    if not body.answers:
        raise HTTPException(400, detail="퀴즈 결과가 비어있습니다.")

    answers_path = ROOT / "input/quiz_answers.json"
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    save_json("input/quiz_answers.json", body.model_dump())

    result = run_script("scripts/ai1_onboarding.py", "--process-result", "input/quiz_answers.json")
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "퀴즈 결과 처리 실패")

    return load_json("output/user_profile.json")


@app.post("/api/onboarding/cat/start")
def cat_start():
    """CAT 온보딩 시작 → 첫 문항 반환"""
    result = run_script("scripts/ai1_onboarding.py", "--cat-start")
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "CAT 시작 실패")
    return load_json("output/cat_response.json")


@app.post("/api/onboarding/cat/answer")
def cat_answer(body: CatAnswer):
    """CAT 한 문항 제출 → 다음 문항 or 완료 반환"""
    save_json("input/cat_answer.json", body.model_dump())
    result = run_script("scripts/ai1_onboarding.py", "--cat-answer", "input/cat_answer.json")
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "CAT 응답 처리 실패")
    return load_json("output/cat_response.json")


@app.get("/api/schedule/today")
def get_today_schedule(
    daily_limit: int = Query(default=100, ge=1, le=1000),
    total_words: int | None = Query(default=None, ge=1),
    days: int | None = Query(default=None, ge=1),
):
    """오늘의 학습 스케줄 반환. total_words+days 모두 있으면 daily_limit 자동 계산."""
    if total_words is not None and days is not None:
        daily_new = total_words / days
        daily_limit = round(daily_new / (1 - 0.4))
    result = run_script(
        "scripts/ai4_scheduler.py",
        "--today-only",
        "--daily-limit", str(daily_limit),
    )
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "스케줄 생성 실패")
    return load_json("output/daily_schedule.json")


@app.post("/api/session/result")
def submit_session_result(body: SessionResult):
    """세션 결과 제출 → FSRS + userRating 업데이트"""
    if not body.answers:
        raise HTTPException(400, detail="세션 결과가 비어있습니다.")

    save_json("input/session_result.json", body.model_dump())

    result = run_script(
        "scripts/ai4_scheduler.py",
        "--submit-result", "input/session_result.json",
    )
    if result.returncode != 0:
        raise HTTPException(500, detail=result.stderr or "세션 처리 실패")

    return load_json("output/user_profile.json")


@app.get("/api/user/profile")
def get_user_profile():
    """유저 프로필 반환"""
    if not (ROOT / "output/user_profile.json").exists():
        raise HTTPException(404, detail="온보딩 미완료")
    return load_json("output/user_profile.json")


@app.get("/api/words/all")
def get_all_words():
    """전체 단어 + 레이팅 반환"""
    return load_json("output/rated_words.json")


@app.post("/api/session/checkpoint")
def session_checkpoint(body: SessionCheckpoint):
    """세션 중간 정답률 → 남은 단어풀 난이도 재정렬 + user_rating 실시간 업데이트"""
    if not body.answers:
        raise HTTPException(400, detail="answers가 비어있습니다.")

    accuracy = sum(1 for a in body.answers if a.correct) / len(body.answers)
    profile = load_json("output/user_profile.json")
    rated = load_json("output/rated_words.json")

    user_rating = profile["user_rating"]
    word_rating = {w["word"]: w["rating"] for w in rated["words"]}
    remaining = list(body.remaining_words)

    # IRT 실시간 업데이트 + checkpoint_answered 누적 저장
    k = _get_k(profile.get("total_sessions", 0))
    newly_answered = []
    for a in body.answers:
        if a.correct is not None:
            wr = word_rating.get(a.word, user_rating)
            user_rating = _irt_update(user_rating, wr, a.correct, k)
            newly_answered.append(a.word)
    profile["user_rating"] = user_rating
    existing = profile.get("checkpoint_answered", [])
    profile["checkpoint_answered"] = list(set(existing) | set(newly_answered))
    save_json("output/user_profile.json", profile)

    if accuracy > 0.75:
        # 어려운 단어 우선 (userRating+100 이상 앞으로)
        hard = [w for w in remaining if word_rating.get(w, user_rating) >= user_rating + 100]
        rest = [w for w in remaining if word_rating.get(w, user_rating) < user_rating + 100]
        remaining = hard + rest
        mode = "hard"
    elif accuracy < 0.50:
        # 쉬운 단어 우선 + 미완료 복습 단어 추가
        easy = [w for w in remaining if word_rating.get(w, user_rating) <= user_rating - 100]
        rest = [w for w in remaining if word_rating.get(w, user_rating) > user_rating - 100]
        remaining = easy + rest

        today_str = datetime.now().date().isoformat()
        answered = {a.word for a in body.answers}
        extra = [
            w["word"] for w in rated["words"]
            if w.get("learned") and w.get("fsrs") and w["fsrs"].get("due_date")
            and w["fsrs"]["due_date"][:10] <= today_str
            and w["word"] not in set(body.remaining_words)
            and w["word"] not in answered
        ][:10]
        remaining = remaining + extra
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


@app.get("/api/recommend/words")
def recommend_words(count: int = Query(default=10, ge=1, le=50)):
    """Oxford DB에서 유저 수준 근접 미학습 단어 추천 (약점 구간 가중치)"""
    if not (ROOT / "output/user_profile.json").exists():
        raise HTTPException(404, detail="온보딩 미완료")
    if not (ROOT / "models/refined_db.json").exists():
        raise HTTPException(404, detail="Oxford DB 없음 (AI3 미실행)")

    profile = load_json("output/user_profile.json")
    rated = load_json("output/rated_words.json")
    db = load_json("models/refined_db.json")

    user_rating = profile["user_rating"]
    all_user_words = {w["word"] for w in rated["words"]}

    # 약점 구간: Again(rating=1) 응답이 많은 단어들의 rating 평균
    weak_ratings = [
        w["rating"] for w in rated["words"]
        for r in w.get("review_history", [])
        if r[1] == 1
    ]
    weak_center = int(sum(weak_ratings) / len(weak_ratings)) if weak_ratings else user_rating

    candidates = [
        w for w in db["words"]
        if w["word"] not in all_user_words
        and abs(w["rating_refined"] - user_rating) <= 150
    ]

    def _score(w):
        proximity = abs(w["rating_refined"] - user_rating)
        weak_boost = max(0, 100 - abs(w["rating_refined"] - weak_center))
        return proximity - weak_boost

    candidates.sort(key=_score)

    return {
        "user_rating": user_rating,
        "weak_center": weak_center,
        "count": min(count, len(candidates)),
        "words": [
            {"word": w["word"], "rating": w["rating_refined"], "pos": w.get("pos")}
            for w in candidates[:count]
        ],
    }


@app.post("/api/recommend/ai")
def recommend_ai(body: AiRecommendRequest):
    """Claude API로 유저 맞춤 단어 추천 → rated_words.json 자동 편입"""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, detail="ANTHROPIC_API_KEY 없음")
    if not (ROOT / "output/user_profile.json").exists():
        raise HTTPException(404, detail="온보딩 미완료")

    profile = load_json("output/user_profile.json")
    rated = load_json("output/rated_words.json")

    user_rating = profile["user_rating"]
    learned_words = [w["word"] for w in rated["words"] if w.get("learned")]

    weak_ratings = [
        w["rating"] for w in rated["words"]
        for r in w.get("review_history", []) if r[1] == 1
    ]
    weak_range = None
    if weak_ratings:
        wc = int(sum(weak_ratings) / len(weak_ratings))
        weak_range = [wc - 100, wc + 100]

    cefr = ("A1" if user_rating < 175 else "A2" if user_rating < 325 else
            "B1" if user_rating < 475 else "B2" if user_rating < 625 else "C1")

    user_ctx = {
        "user_rating": user_rating,
        "cefr_level": cefr,
        "domain": body.domain,
        "weak_rating_range": weak_range,
        "recently_learned": learned_words[-20:],
        "count": body.count,
    }

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="당신은 영어 단어 학습 추천 전문가입니다. 반드시 JSON만 반환하세요.",
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
    raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            raise HTTPException(500, detail="Claude 응답 파싱 실패")

    all_user_words = {w["word"] for w in rated["words"]}
    added = []
    for item in result.get("recommended", []):
        word = item.get("word", "").lower().strip()
        if not word or word in all_user_words:
            continue
        rated["words"].append({
            "word": word,
            "pos": None,
            "meaning": None,
            "rating": item.get("rating", user_rating),
            "source": "ai_recommended",
            "confidence": 0.8,
            "learned": False,
            "fsrs": {
                "stability": None, "difficulty": None, "due_date": None,
                "review_count": 0, "last_rating": None, "state": "new",
                "first_exposure": False,
            },
        })
        all_user_words.add(word)
        added.append({"word": word, "rating": item.get("rating", user_rating),
                      "reason": item.get("reason"), "cefr": item.get("cefr")})

    if added:
        rated["total_words"] = len(rated["words"])
        save_json("output/rated_words.json", rated)

    return {"added_count": len(added), "words": added}


# ── 단독 실행 테스트 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FastAPI 서버")
    parser.add_argument("--test", action="store_true", help="health 엔드포인트 단독 테스트")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.test:
        print("[SERVER TEST] 상태 확인")
        checks = {
            "models/refined_db.json": (ROOT / "models/refined_db.json").exists(),
            "output/rated_words.json": (ROOT / "output/rated_words.json").exists(),
            "output/user_profile.json": (ROOT / "output/user_profile.json").exists(),
            "output/daily_schedule.json": (ROOT / "output/daily_schedule.json").exists(),
        }
        for path, ok in checks.items():
            print(f"  {'✓' if ok else '✗'} {path}")
        print("[SERVER TEST] 완료. 서버 실행:")
        print(f"  uvicorn unity_bridge.server:app --host {args.host} --port {args.port} --reload")
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
