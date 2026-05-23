"""
FastAPI Unity 브릿지 서버
uvicorn unity_bridge.server:app --host 0.0.0.0 --port 8000 --reload
"""
import json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 스크립트 루트를 이 파일 기준으로 결정 (unity_bridge/ 의 부모)
ROOT = Path(__file__).parent.parent

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
    """프로젝트 루트에서 scripts/ 스크립트 실행."""
    cmd = [sys.executable] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)


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

    result = run_script("scripts/ai2_rate_csv.py", "--input", "input/user_words.csv", timeout=120)
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
def get_today_schedule(daily_limit: int = Query(default=100, ge=1, le=1000)):
    """오늘의 학습 스케줄 반환"""
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
