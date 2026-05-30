"""
AI2 — 유저 CSV 미등록 단어 레이팅 예측
Oxford DB 임베딩 KNN + (저신뢰 시) Claude API 검증.
"""
import os, sys, json, math, logging, argparse, pickle, re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/error.log", encoding="utf-8"),
    ],
    format="%(asctime)s [AI2] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

EMPTY_FSRS = {
    "stability": None,
    "difficulty": None,
    "due_date": None,
    "review_count": 0,
    "last_rating": None,
    "state": "new",
    "first_exposure": False,
}


# ── KNN 예측 ─────────────────────────────────────────────────────────────────

def knn_predict(words: list[str], cache: dict, k: int = 5) -> list[dict]:
    st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = st_model.encode(words, batch_size=64, show_progress_bar=len(words) > 20)

    db_embeddings = cache["embeddings"]  # (N, 384)
    db_ratings = np.array(cache["ratings"])

    results = []
    for word, emb in zip(words, embeddings):
        sims = cosine_similarity([emb], db_embeddings)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        top_sims = sims[top_idx]
        top_ratings = db_ratings[top_idx]

        weight_sum = top_sims.sum()
        if weight_sum == 0:
            predicted = float(np.mean(top_ratings))
        else:
            predicted = float(np.dot(top_sims, top_ratings) / weight_sum)

        std = float(np.std(top_ratings))
        confidence = max(0.5, 1.0 - std / 300.0)

        results.append(
            {
                "word": word,
                "predicted_rating": round(predicted),
                "std": round(std, 2),
                "confidence": round(confidence, 3),
                "low_confidence": confidence < 0.7 or std > 150,
            }
        )
    return results


# ── JSON 복구 파서 ───────────────────────────────────────────────────────────

def _parse_json_robust(raw: str) -> dict | None:
    """Claude 응답 JSON을 단계적으로 복구 시도. 실패 시 None 반환."""
    def _clean(s: str) -> str:
        s = re.sub(r"//[^\n]*", "", s)           # // 주석 제거
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # /* */ 주석 제거
        s = re.sub(r",\s*([\}\]])", r"\1", s)    # trailing comma 제거
        return s.strip()

    # 1차: 그대로 파싱
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2차: 주석·trailing comma 제거 후 파싱
    try:
        return json.loads(_clean(raw))
    except json.JSONDecodeError:
        pass

    # 3차: 외부 {...} 추출 후 정리하여 파싱
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(_clean(m.group()))
        except json.JSONDecodeError:
            pass

    return None


# ── Claude API 저신뢰 검증 ───────────────────────────────────────────────────

def api_verify(low_conf_words: list[dict], client: anthropic.Anthropic, batch_size: int = 50) -> dict:
    verified: dict = {}
    total_batches = math.ceil(len(low_conf_words) / batch_size)

    for i in range(0, len(low_conf_words), batch_size):
        batch = low_conf_words[i : i + batch_size]
        batch_no = i // batch_size + 1
        logger.info(f"저신뢰 검증 배치 {batch_no}/{total_batches} ({len(batch)}단어)")

        knn_ref = {w["word"]: w["predicted_rating"] for w in batch}
        word_list = [w["word"] for w in batch]

        prompt = (
            "다음 단어들의 CEFR 레벨을 판단하고 레이팅을 부여하세요.\n"
            "레이팅 기준: A1=100, A2=250, B1=400, B2=550, C1=700, C2(초고급)=850\n\n"
            f"단어 목록: {json.dumps(word_list, ensure_ascii=False)}\n\n"
            f"KNN 예측값(참고용): {json.dumps(knn_ref, ensure_ascii=False)}\n\n"
            '응답 형식 (JSON만):\n'
            '{"word": {"rating": 700, "cefr": "C1", "reason": "고급 추상 형용사"}, ...}'
        )

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=(
                    "당신은 영어 단어 난이도 평가 전문가입니다. "
                    "Oxford 3000/5000 기준으로 CEFR 레벨을 판단합니다. "
                    "반드시 JSON만 반환하세요."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # 마크다운 코드블록 제거
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            batch_result = _parse_json_robust(raw)
            if batch_result is None:
                raise ValueError("JSON 파싱 실패 (모든 복구 시도 실패)")
            verified.update(batch_result)
        except Exception as exc:
            logger.error(f"API 검증 배치 {batch_no} 실패: {exc}")

    return verified


# ── IQR 이상치 탐지 ──────────────────────────────────────────────────────────

def flag_outliers(words_data: list[dict]) -> None:
    ratings = [w["rating"] for w in words_data]
    if len(ratings) < 4:
        return
    q1, q3 = np.percentile(ratings, 25), np.percentile(ratings, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    for w in words_data:
        if w["rating"] < lo or w["rating"] > hi:
            logger.warning(f"이상치 단어: {w['word']} (rating={w['rating']})")


# ── 메인 ────────────────────────────────────────────────────────────────────

def main(csv_path: str) -> None:
    refined_path = "models/refined_db.json"
    embed_path = "models/embeddings_cache.pkl"

    if not os.path.exists(refined_path) or not os.path.exists(embed_path):
        logger.error("AI3 먼저 실행하세요 (models/refined_db.json, embeddings_cache.pkl 없음)")
        sys.exit(1)

    if not os.path.exists(csv_path):
        logger.error(f"CSV 파일 없음: {csv_path}")
        sys.exit(1)

    # 1. CSV 로드 (인코딩 자동 감지)
    df_csv = None
    for enc in ("utf-8-sig", "utf-8", "euc-kr", "cp949"):
        try:
            df_csv = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df_csv is None:
        logger.error("CSV 인코딩을 인식할 수 없습니다.")
        sys.exit(1)

    # WORD 컬럼 탐색: 영문 WORD 또는 한글 '단어' 허용
    col_map = {c.strip().upper(): c for c in df_csv.columns}
    word_col = col_map.get("WORD") or col_map.get("단어".upper())
    if word_col is None:
        # fallback: '단어' 직접 탐색
        for c in df_csv.columns:
            if c.strip() in ("단어", "WORD", "word"):
                word_col = c
                break
    if word_col is None:
        logger.error(f"WORD/단어 컬럼 없음. 실제 컬럼: {list(df_csv.columns)}")
        sys.exit(1)

    # 의미 컬럼 탐색 (있으면 활용)
    meaning_col = col_map.get("MEANING") or col_map.get("뜻".upper())
    if meaning_col is None:
        for c in df_csv.columns:
            if c.strip() in ("뜻", "MEANING", "meaning"):
                meaning_col = c
                break

    df_csv["_WORD"] = df_csv[word_col].astype(str).str.lower().str.strip()
    df_csv["_WORD"] = df_csv["_WORD"].str.replace(r"[^\w]", "", regex=True)
    df_csv = df_csv[df_csv["_WORD"].str.len() > 0].drop_duplicates(subset=["_WORD"])
    user_words = df_csv["_WORD"].tolist()

    # 의미 매핑 (유저 CSV에 한국어 뜻이 있으면 predicted 단어에도 활용)
    user_meaning_map: dict = {}
    if meaning_col:
        for _, row in df_csv.iterrows():
            m = row.get(meaning_col)
            if pd.notna(m) and str(m).strip():
                user_meaning_map[row["_WORD"]] = str(m).strip()

    if not user_words:
        logger.error("CSV가 비어있습니다.")
        sys.exit(1)

    logger.info(f"CSV 로드: {len(user_words)}단어")

    # 2. Oxford DB 로드
    with open(refined_path, "r", encoding="utf-8") as f:
        refined_db = json.load(f)

    oxford_map: dict = {w["word"]: w for w in refined_db["words"]}

    # 3. 매칭 / 미매칭 분리
    matched, unmatched = [], []
    for word in user_words:
        if word in oxford_map:
            ox = oxford_map[word]
            matched.append(
                {
                    "word": word,
                    "pos": ox.get("pos"),
                    "meaning": ox.get("meaning"),
                    "rating": ox["rating_refined"],
                    "source": "oxford_db",
                    "confidence": 1.0,
                    "learned": False,
                    "fsrs": dict(EMPTY_FSRS),
                }
            )
        else:
            unmatched.append(word)

    logger.info(f"Oxford 매칭: {len(matched)}, 미등록: {len(unmatched)}")
    if len(unmatched) > len(user_words) * 0.2:
        logger.warning(f"미등록 단어 비율 {len(unmatched)/len(user_words)*100:.1f}% > 20%")

    # 4. KNN 예측 + 저신뢰 분리
    predicted_words = []
    api_verified_count = 0

    if unmatched:
        with open(embed_path, "rb") as f:
            cache = pickle.load(f)

        knn_results = knn_predict(unmatched, cache)
        low_conf = [r for r in knn_results if r["low_confidence"]]

        # 5. Claude API 검증
        api_result: dict = {}
        if low_conf:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                client = anthropic.Anthropic(api_key=api_key)
                api_result = api_verify(low_conf, client)
            else:
                logger.warning("ANTHROPIC_API_KEY 없음. KNN 결과만 사용.")

        for r in knn_results:
            word = r["word"]
            if word in api_result:
                rating = api_result[word].get("rating", r["predicted_rating"])
                source = "api_verified"
                confidence = 0.9
                api_verified_count += 1
            else:
                rating = r["predicted_rating"]
                source = "predicted"
                confidence = r["confidence"]

            predicted_words.append(
                {
                    "word": word,
                    "pos": None,
                    "meaning": user_meaning_map.get(word),
                    "rating": int(rating),
                    "source": source,
                    "confidence": round(confidence, 3),
                    "learned": False,
                    "fsrs": dict(EMPTY_FSRS),
                }
            )

    all_words = matched + predicted_words
    ratings = [w["rating"] for w in all_words]
    flag_outliers(all_words)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "total_words": len(all_words),
        "oxford_matched": len(matched),
        "predicted": len(predicted_words) - api_verified_count,
        "api_verified": api_verified_count,
        "stats": {
            "mean_rating": round(float(np.mean(ratings)), 1) if ratings else 0,
            "median_rating": round(float(np.median(ratings)), 1) if ratings else 0,
            "std_rating": round(float(np.std(ratings)), 1) if ratings else 0,
        },
        "words": all_words,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/rated_words.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"rated_words.json 저장: {len(all_words)}단어")
    print(f"[AI2] 완료! 전체 {len(all_words)}단어 (Oxford:{len(matched)}, 예측:{len(predicted_words)})")


# ── 단독 테스트 ─────────────────────────────────────────────────────────────

def run_test() -> None:
    print("[AI2 TEST] 단독 테스트")

    # KNN confidence 공식 검증
    ratings_sample = np.array([550, 700, 680, 600, 720])
    std = float(np.std(ratings_sample))
    conf = max(0.5, 1.0 - std / 300.0)
    print(f"  KNN std={std:.1f}, confidence={conf:.3f}, low_conf={conf < 0.7}")

    # 이상치 탐지 테스트
    mock_words = [
        {"word": "apple", "rating": 100},
        {"word": "negotiate", "rating": 550},
        {"word": "abandon", "rating": 700},
        {"word": "outlier_word", "rating": 1500},
    ]
    flag_outliers(mock_words)
    print("  이상치 탐지 실행 완료 (위 WARNING 확인)")

    if os.path.exists("models/refined_db.json") and os.path.exists("models/embeddings_cache.pkl"):
        print("  AI3 결과물 발견. 실제 Oxford 매칭 스킵 (--input 으로 실행 가능)")
    else:
        print("  AI3 미실행. KNN 테스트 생략.")

    print("[AI2 TEST] 완료")


def default_only() -> None:
    """CSV 없이 빈 rated_words.json 생성. Oxford DB 보충은 AI4 스케줄러가 담당."""
    refined_path = "models/refined_db.json"
    if not os.path.exists(refined_path):
        logger.error("AI3 먼저 실행하세요 (models/refined_db.json 없음)")
        sys.exit(1)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "total_words": 0,
        "oxford_matched": 0,
        "predicted": 0,
        "api_verified": 0,
        "stats": {"mean_rating": 0, "median_rating": 0, "std_rating": 0},
        "words": [],
    }
    os.makedirs("output", exist_ok=True)
    with open("output/rated_words.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("CSV 없음 — 빈 rated_words.json 생성 완료.")
    print("[AI2] 기본 단어 DB 생성 완료 (단어 0개, Oxford 보충은 스케줄러가 처리)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI2: 유저 CSV 레이팅 예측")
    parser.add_argument("--input", default="input/user_words.csv", help="입력 CSV 경로")
    parser.add_argument("--default-only", action="store_true", help="CSV 없이 빈 rated_words.json 생성")
    parser.add_argument("--test", action="store_true", help="단독 기능 테스트")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.default_only:
        default_only()
    else:
        main(args.input)
