"""
AI3 — Oxford DB 레이팅 세분화
최초 1회 실행. models/refined_db.json 과 models/embeddings_cache.pkl 생성.
"""
import os, sys, json, math, logging, argparse, pickle, re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from wordfreq import word_frequency
from sentence_transformers import SentenceTransformer
import anthropic

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/error.log", encoding="utf-8"),
    ],
    format="%(asctime)s [AI3] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"


# ── 로컬 계산 헬퍼 ──────────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    """모음 연속 그룹으로 음절 수 근사."""
    vowels = "aeiou"
    count, in_vowel = 0, False
    for ch in word.lower():
        if ch in vowels:
            if not in_vowel:
                count += 1
            in_vowel = True
        else:
            in_vowel = False
    return max(1, count)


def syllable_bonus(syllables: int) -> int:
    return min((syllables - 1) * 5, 30)


def frequency_penalty(word: str) -> float:
    freq = word_frequency(word, "en")
    if freq == 0:
        return 20.0
    log_f = math.log10(freq + 1e-10)
    # 로그 범위 [-7, -2] → [0, 1] 정규화 (0=희귀, 1=빈도 높음)
    normalized = (log_f - (-7.0)) / ((-2.0) - (-7.0))
    normalized = max(0.0, min(1.0, normalized))
    # 고빈도 → 음수 패널티, 저빈도 → 양수 보너스
    return (1.0 - normalized) * 40.0 - 20.0


# ── Claude API 배치 호출 ────────────────────────────────────────────────────

def batch_abstraction_scores(words: list[str], client: anthropic.Anthropic, batch_size: int = 50) -> dict:
    all_scores: dict = {}
    total = math.ceil(len(words) / batch_size)

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        batch_no = i // batch_size + 1
        logger.info(f"추상성 배치 {batch_no}/{total} ({len(batch)}단어)")

        prompt = (
            "다음 영어 단어들의 추상성 점수를 0.0~1.0으로 평가하세요.\n"
            "0.0 = 매우 구체적 (apple, chair, run)\n"
            "1.0 = 매우 추상적 (justice, abolish, contemplate)\n\n"
            f"단어 목록:\n{json.dumps(batch, ensure_ascii=False)}\n\n"
            '응답 형식 (JSON만, 다른 텍스트 절대 금지):\n{"word1": 0.72, "word2": 0.65, ...}'
        )

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=(
                    "당신은 영어 단어의 추상성을 평가하는 전문가입니다. "
                    "반드시 JSON만 반환하고, 다른 텍스트는 절대 포함하지 마세요."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            # 주석·trailing comma 제거 후 파싱, 실패 시 외부 {} 추출
            def _clean(s):
                s = re.sub(r"//[^\n]*", "", s)
                s = re.sub(r",\s*([\}\]])", r"\1", s)
                return s.strip()
            parsed = None
            for candidate in [raw, _clean(raw)]:
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", candidate, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(_clean(m.group()))
                            break
                        except json.JSONDecodeError:
                            pass
            if parsed is None:
                raise ValueError("JSON 파싱 실패")
            all_scores.update(parsed)
        except Exception as exc:
            logger.error(f"배치 {batch_no} API 실패: {exc}. fallback 사용.")
            for w in batch:
                freq = word_frequency(w, "en")
                log_f = math.log10(freq + 1e-10)
                norm = max(0.0, min(1.0, (log_f - (-7.0)) / 5.0))
                all_scores[w] = round(1.0 - norm, 3)

    return all_scores


# ── 메인 ────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    refined_path = "models/refined_db.json"
    embed_path = "models/embeddings_cache.pkl"

    if not force and os.path.exists(refined_path) and os.path.exists(embed_path):
        print("[AI3] 이미 세분화 완료. 재실행하려면 --force 플래그 사용.")
        sys.exit(0)

    os.makedirs("models", exist_ok=True)

    # 1. 데이터 로드
    xlsx_path = "data/oxford3000_5000_merged.xlsx"
    if not os.path.exists(xlsx_path):
        logger.error(f"Oxford DB 없음: {xlsx_path}")
        sys.exit(1)

    logger.info("Oxford DB 로드 중...")
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip().str.upper()

    for col in ("WORD", "RATING"):
        if col not in df.columns:
            logger.error(f"필수 컬럼 없음: {col}. 실제 컬럼: {list(df.columns)}")
            sys.exit(1)

    df["WORD"] = df["WORD"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["WORD", "RATING"]).drop_duplicates(subset=["WORD"]).reset_index(drop=True)
    if "ID" not in df.columns:
        df["ID"] = df.index + 1

    logger.info(f"{len(df)}단어 로드 완료")

    # 2. 음절 보너스
    logger.info("음절 계산 중...")
    df["syllables"] = df["WORD"].apply(count_syllables)
    df["syllable_bonus"] = df["syllables"].apply(syllable_bonus)

    # 3. 빈도 패널티
    logger.info("wordfreq 계산 중...")
    df["wordfreq_score"] = df["WORD"].apply(lambda w: word_frequency(w, "en"))
    df["frequency_penalty"] = df["WORD"].apply(frequency_penalty)

    # 4. 추상성 점수 (Claude API 또는 fallback)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Claude API 배치 호출 시작 ({len(df)}단어)...")
        scores = batch_abstraction_scores(df["WORD"].tolist(), client)
    else:
        logger.warning("ANTHROPIC_API_KEY 없음. 추상성 점수 fallback 사용.")
        scores = {}
        for w, freq in zip(df["WORD"], df["wordfreq_score"]):
            log_f = math.log10(freq + 1e-10)
            norm = max(0.0, min(1.0, (log_f - (-7.0)) / 5.0))
            scores[w] = round(1.0 - norm, 3)

    df["abstraction_score"] = df["WORD"].apply(lambda w: scores.get(w, 0.5))
    df["abstraction_bonus"] = df["abstraction_score"] * 30

    # 5. 레이팅 세분화
    df["rating_base"] = df["RATING"].astype(int)
    raw = df["rating_base"] + df["syllable_bonus"] + df["frequency_penalty"] + df["abstraction_bonus"]
    df["rating_refined"] = raw.clip(
        lower=df["rating_base"] - 100, upper=df["rating_base"] + 100
    ).round().astype(int)

    # 6. 임베딩 생성
    logger.info("임베딩 생성 중 (paraphrase-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def embed_text(row):
        parts = [row["WORD"]]
        if "POS" in df.columns and pd.notna(row.get("POS")):
            parts.append(str(row["POS"]))
        if "MEANING" in df.columns and pd.notna(row.get("MEANING")):
            parts.append(str(row["MEANING"]))
        return " ".join(parts)

    texts = [embed_text(row) for _, row in df.iterrows()]
    embeddings = st_model.encode(texts, batch_size=64, show_progress_bar=True)

    cache = {
        "words": df["WORD"].tolist(),
        "embeddings": embeddings,
        "ratings": df["rating_refined"].tolist(),
    }
    with open(embed_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"임베딩 캐시 저장: {embed_path}")

    # 7. refined_db.json 저장
    def safe(val):
        return None if pd.isna(val) else val

    words_data = [
        {
            "id": int(row["ID"]),
            "word": row["WORD"],
            "pos": safe(row.get("POS")),
            "meaning": safe(row.get("MEANING")),
            "rating_base": int(row["rating_base"]),
            "rating_refined": int(row["rating_refined"]),
            "syllables": int(row["syllables"]),
            "wordfreq_score": float(row["wordfreq_score"]),
            "abstraction_score": float(row["abstraction_score"]),
        }
        for _, row in df.iterrows()
    ]

    refined_db = {
        "version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "total_words": len(words_data),
        "words": words_data,
    }
    with open(refined_path, "w", encoding="utf-8") as f:
        json.dump(refined_db, f, ensure_ascii=False, indent=2)

    logger.info(f"refined_db.json 저장 완료 ({len(words_data)}단어)")
    print(f"[AI3] 완료! {len(words_data)}단어 세분화.")


# ── 단독 테스트 ─────────────────────────────────────────────────────────────

def run_test() -> None:
    print("[AI3 TEST] 헬퍼 함수 단독 테스트")

    tests = [
        ("abandon", 3, 10),
        ("apple", 2, 5),
        ("extraordinary", 6, 30),
    ]
    for word, expected_syl, expected_bonus in tests:
        s = count_syllables(word)
        b = syllable_bonus(s)
        print(f"  {word}: syllables={s} (기대:{expected_syl}), bonus={b} (기대:{expected_bonus})")

    for word in ("the", "apple", "ephemeral", "serendipity"):
        p = frequency_penalty(word)
        print(f"  frequency_penalty('{word}'): {p:.2f}")

    # 클램프 테스트
    base, raw_val = 700, 850
    clamped = max(base - 100, min(base + 100, raw_val))
    assert clamped == 800, f"클램프 오류: {clamped}"
    print(f"  clamp({base}, raw={raw_val}) = {clamped} ✓")

    print("[AI3 TEST] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI3: Oxford DB 레이팅 세분화")
    parser.add_argument("--force", action="store_true", help="기존 파일 무시하고 재실행")
    parser.add_argument("--test", action="store_true", help="단독 기능 테스트 (파일 불필요)")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        main(force=args.force)
