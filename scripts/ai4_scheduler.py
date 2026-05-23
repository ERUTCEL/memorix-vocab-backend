"""
AI4 — 일별 학습 스케줄러 (FSRS + IRT)
--today-only --daily-limit N  : 오늘 스케줄 생성 → daily_schedule.json
--days N    --daily-limit N   : N일 플랜 (오늘 스케줄 저장)
--submit-result <path>        : 세션 결과 처리 → FSRS + userRating 업데이트
"""
import os, sys, json, math, logging, argparse
from datetime import datetime, timedelta, date
from scipy.optimize import minimize

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/error.log", encoding="utf-8"),
    ],
    format="%(asctime)s [AI4] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FSRS 상수 ────────────────────────────────────────────────────────────────

DECAY = -0.5
FACTOR = 0.9 ** (1.0 / DECAY) - 1  # ≈ 0.2315
FSRS_WEIGHTS = [
    0.4072, 1.1829, 3.1262, 15.4722,
    7.2102, 0.5316, 1.0651, 0.0589,
    1.5330, 0.1544, 1.0070,
    1.9395, 0.1100, 0.2900,
    2.2700, 0.0600, 2.1700,
    0.0800, 0.0100, 0.3400, 1.3,
]
MAX_STABILITY = 36500.0  # 100년 클램프


# ── FSRS 핵심 함수 ───────────────────────────────────────────────────────────

def retrievability(t: float, S: float) -> float:
    """t: 경과일, S: stability"""
    if S <= 0:
        return 0.0
    return (1.0 + FACTOR * t / S) ** DECAY


def initial_stability(rating: int) -> float:
    return FSRS_WEIGHTS[rating - 1]


def initial_difficulty(rating: int) -> float:
    return FSRS_WEIGHTS[4] - (rating - 3) * FSRS_WEIGHTS[5]


def stability_after_recall(D: float, S: float, R: float, rating: int) -> float:
    hard_penalty = FSRS_WEIGHTS[15] if rating == 2 else 1.0
    easy_bonus = FSRS_WEIGHTS[16] if rating == 4 else 1.0
    new_S = S * (
        math.exp(FSRS_WEIGHTS[8])
        * (11.0 - D)
        * (S ** -FSRS_WEIGHTS[9])
        * (math.exp(FSRS_WEIGHTS[10] * (1.0 - R)) - 1.0)
        * hard_penalty
        * easy_bonus
    ) + S
    return min(max(new_S, 0.01), MAX_STABILITY)


def stability_after_forget(D: float, S: float, R: float) -> float:
    new_S = (
        FSRS_WEIGHTS[11]
        * (D ** -FSRS_WEIGHTS[12])
        * ((S + 1.0) ** FSRS_WEIGHTS[13] - 1.0)
        * math.exp(FSRS_WEIGHTS[14] * (1.0 - R))
    )
    return min(max(new_S, 0.01), MAX_STABILITY)


def update_difficulty(D: float, rating: int) -> float:
    delta = FSRS_WEIGHTS[6] * (rating - 3)
    new_D = D - delta
    # mean reversion
    new_D = FSRS_WEIGHTS[7] * initial_difficulty(3) + (1.0 - FSRS_WEIGHTS[7]) * new_D
    return max(1.0, min(10.0, new_D))


def next_interval(S: float, target_r: float = 0.9) -> int:
    interval = S * (target_r ** (1.0 / DECAY) - 1.0) / FACTOR
    return max(1, round(interval))


# ── 카드 상태 처리 ───────────────────────────────────────────────────────────

def init_fsrs_card(rating: int, today: date) -> dict:
    S = initial_stability(rating)
    D = initial_difficulty(rating)
    if rating == 1:
        state = "learning"
        due = (datetime.combine(today, datetime.min.time()) + timedelta(minutes=10)).isoformat()
    else:
        state = "review"
        interval = next_interval(S)
        due = (today + timedelta(days=interval)).isoformat()
    return {
        "stability": round(S, 4),
        "difficulty": round(D, 4),
        "due_date": due,
        "review_count": 1,
        "last_rating": rating,
        "state": state,
        "last_review": today.isoformat(),
        "first_exposure": True,
    }


def process_review(fsrs: dict, rating: int, today: date) -> dict:
    last_review = date.fromisoformat(fsrs["last_review"][:10])
    t = (today - last_review).days
    R = retrievability(t, fsrs["stability"])
    D = fsrs["difficulty"]
    S = fsrs["stability"]

    if rating == 1:
        new_S = stability_after_forget(D, S, R)
        state = "relearning"
        due = (today + timedelta(days=1)).isoformat()
    else:
        new_S = stability_after_recall(D, S, R, rating)
        state = "review"
        interval = next_interval(new_S)
        due = (today + timedelta(days=interval)).isoformat()

    new_D = update_difficulty(D, rating)

    return {
        "stability": round(new_S, 4),
        "difficulty": round(new_D, 4),
        "due_date": due,
        "review_count": fsrs["review_count"] + 1,
        "last_rating": rating,
        "state": state,
        "last_review": today.isoformat(),
        "first_exposure": False,
    }


# ── IRT 유틸 ────────────────────────────────────────────────────────────────

def get_k_factor(total_sessions: int) -> int:
    if total_sessions <= 5:
        return 100
    elif total_sessions <= 20:
        return 50
    return 20


def update_user_rating(user_rating: int, word_rating: int, correct: bool, k: int) -> int:
    expected = 1.0 / (1.0 + math.exp(-(user_rating - word_rating) / 150.0))
    actual = 1 if correct else 0
    return round(user_rating + k * (actual - expected))


# ── 파일 유틸 ────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Optimizer 데이터 ─────────────────────────────────────────────────────────

def get_optimizer_reviews(rated_words: dict) -> list:
    """첫 노출(first_exposure=True) 제거 후 optimizer에 넘길 카드 목록 반환.
    optimizer는 실제 복습 이력이 있는 카드만 학습해야 함."""
    return [
        w for w in rated_words["words"]
        if w.get("learned") and w.get("fsrs")
        and not w["fsrs"].get("first_exposure", False)
    ]


# ── FSRS Weight Optimizer ────────────────────────────────────────────────────

def _sim_stability(history: list, w: list) -> list[tuple]:
    """카드 리뷰 이력으로 각 리뷰 시점의 (R_predicted, correct) 생성."""
    if len(history) < 2:
        return []
    initial_rating = history[0][1]
    S = max(0.01, w[initial_rating - 1])
    D = max(1.0, min(10.0, w[4] - (initial_rating - 3) * w[5]))
    results = []
    for elapsed, rating in history[1:]:
        R = retrievability(elapsed, S)
        correct = rating > 1
        results.append((R, correct))
        if rating == 1:
            new_S = (
                w[11] * (D ** -w[12])
                * ((S + 1.0) ** w[13] - 1.0)
                * math.exp(w[14] * (1.0 - R))
            )
        else:
            hp = w[15] if rating == 2 else 1.0
            eb = w[16] if rating == 4 else 1.0
            new_S = S * (
                math.exp(w[8]) * (11.0 - D) * (S ** -w[9])
                * (math.exp(w[10] * (1.0 - R)) - 1.0)
                * hp * eb
            ) + S
        S = min(max(new_S, 0.01), MAX_STABILITY)
        delta = w[6] * (rating - 3)
        new_D = D - delta
        new_D = w[7] * (w[4] - 0 * w[5]) + (1.0 - w[7]) * new_D
        D = max(1.0, min(10.0, new_D))
    return results


def _fsrs_loss(w_flat: list, histories: list) -> float:
    """FSRS BCE 손실: Σ -[correct*log(R) + (1-correct)*log(1-R)]"""
    w = list(w_flat)
    total, count = 0.0, 0
    for history in histories:
        for R, correct in _sim_stability(history, w):
            R = max(1e-9, min(1 - 1e-9, R))
            total += -(math.log(R) if correct else math.log(1 - R))
            count += 1
    return total / max(count, 1)


def optimize_weights(rated_words_path: str = "output/rated_words.json", min_cards: int = 30) -> list:
    """리뷰 이력 기반 gradient descent로 FSRS_WEIGHTS 최적화."""
    if not os.path.exists(rated_words_path):
        logger.error("rated_words.json 없음")
        return list(FSRS_WEIGHTS)

    with open(rated_words_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    histories = [
        w["review_history"] for w in data["words"]
        if len(w.get("review_history", [])) >= 2
    ]

    if len(histories) < min_cards:
        logger.warning(f"최적화 데이터 부족: {len(histories)}개 (최소 {min_cards}개 필요)")
        return list(FSRS_WEIGHTS)

    logger.info(f"FSRS 최적화 시작: {len(histories)}개 카드")

    bounds = [
        (0.1, 5.0), (0.5, 5.0), (1.0, 10.0), (5.0, 25.0),  # w[0]-w[3]
        (3.0, 10.0), (0.1, 3.0),                              # w[4]-w[5]
        (0.1, 5.0), (0.01, 0.5),                              # w[6]-w[7]
        (0.5, 5.0), (0.01, 1.0), (0.1, 5.0),                 # w[8]-w[10]
        (0.5, 5.0), (0.01, 1.0), (0.01, 1.0),                # w[11]-w[13]
        (1.0, 5.0), (0.01, 0.5), (1.0, 5.0),                 # w[14]-w[16]
    ]

    result = minimize(
        _fsrs_loss,
        x0=list(FSRS_WEIGHTS[:17]),
        args=(histories,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-7},
    )

    new_weights = list(result.x) + list(FSRS_WEIGHTS[17:])
    if result.success:
        logger.info(f"최적화 완료: loss={result.fun:.4f}")
    else:
        logger.warning(f"수렴 불완전: {result.message}")

    return new_weights


# ── Oxford DB 보충 ───────────────────────────────────────────────────────────

def get_oxford_supplement(user_rating: int, count: int, exclude_words: set) -> list:
    refined_path = "models/refined_db.json"
    if not os.path.exists(refined_path) or count <= 0:
        return []
    db = load_json(refined_path)
    candidates = [
        w for w in db["words"]
        if w["word"] not in exclude_words and abs(w["rating_refined"] - user_rating) <= 150
    ]
    candidates.sort(key=lambda w: abs(w["rating_refined"] - user_rating))
    return candidates[:count]


# ── 복습:신규 비율 계산 ──────────────────────────────────────────────────────

def calculate_review_ratio(user_profile: dict, review_due_count: int) -> float:
    """복습 비율(0.0~1.0) 반환. 기본 40:60, 상황별 자동 조정."""
    if review_due_count == 0 or user_profile.get("total_sessions", 0) == 0:
        return 0.0  # 첫날 또는 복습 없음 → 신규만
    if review_due_count >= 50:
        return 0.6  # 복습 밀림 → 60:40
    last_acc = user_profile.get("last_session_accuracy")
    if last_acc is not None and last_acc < 0.50:
        return 0.5  # 최근 정답률 저조 → 50:50
    return 0.4  # 기본 40:60


# ── 일별 스케줄 생성 ─────────────────────────────────────────────────────────

def build_daily_schedule(rated_words: dict, user_profile: dict, daily_limit: int, today: date) -> dict:
    user_rating = user_profile["user_rating"]
    words = rated_words["words"]
    today_str = today.isoformat()

    # 복습 단어: due_date <= today, learned=True — retrievability 낮은 순(망각 위험 높은 것 먼저)
    review_due = sorted(
        [
            w for w in words
            if w.get("learned") and w.get("fsrs") and w["fsrs"].get("due_date")
            and w["fsrs"]["due_date"][:10] <= today_str
        ],
        key=lambda w: retrievability(
            max(0, (today - date.fromisoformat(
                w["fsrs"].get("last_review", today.isoformat())[:10]
            )).days),
            w["fsrs"].get("stability") or 1.0,
        ),
    )

    review_ratio = calculate_review_ratio(user_profile, len(review_due))
    review_target = int(daily_limit * review_ratio)
    if len(review_due) > daily_limit:
        logger.warning(f"복습 단어({len(review_due)}) > daily_limit({daily_limit}). 신규 0.")
        review_target = daily_limit

    review_words = review_due[:review_target]
    new_target = daily_limit - len(review_words)

    # 신규 단어: userRating 근접 우선
    unlearned = [w for w in words if not w.get("learned")]
    unlearned.sort(key=lambda w: abs(w["rating"] - user_rating))
    new_from_csv = unlearned[:new_target]

    # Oxford DB 보충
    supplement_needed = new_target - len(new_from_csv)
    exclude = {w["word"] for w in words}
    supplement = get_oxford_supplement(user_rating, supplement_needed, exclude)

    schedule = {
        "date": today_str,
        "user_rating": user_rating,
        "total_words": len(review_words) + len(new_from_csv) + len(supplement),
        "new_words": [{"word": w["word"], "rating": w["rating"], "type": "new"} for w in new_from_csv],
        "review_words": [
            {"word": w["word"], "rating": w["rating"], "type": "review", "fsrs_due": w["fsrs"]["due_date"]}
            for w in review_words
        ],
        "db_supplement": [
            {"word": w["word"], "rating": w["rating_refined"], "type": "supplement"}
            for w in supplement
        ],
        "stats": {
            "new_count": len(new_from_csv),
            "review_count": len(review_words),
            "supplement_count": len(supplement),
        },
    }
    return schedule


# ── 세션 결과 처리 ───────────────────────────────────────────────────────────

def process_session_result(session_path: str) -> dict:
    for path, label in [
        ("output/rated_words.json", "rated_words.json"),
        ("output/user_profile.json", "user_profile.json"),
        (session_path, session_path),
    ]:
        if not os.path.exists(path):
            logger.error(f"{label} 없음.")
            sys.exit(1)

    rated_words = load_json("output/rated_words.json")
    user_profile = load_json("output/user_profile.json")
    session = load_json(session_path)
    today = date.today()

    word_map = {w["word"]: w for w in rated_words["words"]}
    k = get_k_factor(user_profile["total_sessions"])
    correct_count = 0
    total_correct_tracked = 0

    for result in session.get("answers", []):
        word = result.get("word", "").lower().strip()
        rating_given = int(result.get("rating_given", 3))
        correct = result.get("correct")

        if word not in word_map:
            logger.warning(f"세션 결과에 없는 단어: {word}")
            continue

        card = word_map[word]
        if not card.get("learned"):
            card["fsrs"] = init_fsrs_card(rating_given, today)
            card["learned"] = True
            card["review_history"] = [[0, rating_given]]
        elif card["fsrs"].get("state") == "learning":
            # 학습 단계 재제출 — full review 공식 대신 init으로 재평가
            elapsed = (today - date.fromisoformat(
                card["fsrs"]["last_review"][:10]
            )).days
            prev_count = card["fsrs"]["review_count"]
            card["fsrs"] = init_fsrs_card(rating_given, today)
            card["fsrs"]["review_count"] = prev_count + 1
            card["fsrs"]["first_exposure"] = False
            card.setdefault("review_history", []).append([elapsed, rating_given])
        else:
            elapsed = (today - date.fromisoformat(
                card["fsrs"]["last_review"][:10]
            )).days
            card["fsrs"] = process_review(card["fsrs"], rating_given, today)
            card.setdefault("review_history", []).append([elapsed, rating_given])

        if correct is not None:
            total_correct_tracked += 1
            if correct:
                correct_count += 1
            user_profile["user_rating"] = update_user_rating(
                user_profile["user_rating"], card["rating"], correct, k
            )

    user_profile["total_sessions"] += 1
    user_profile["k_factor"] = get_k_factor(user_profile["total_sessions"])
    user_profile["last_updated"] = today.isoformat()
    user_profile["rating_history"].append(user_profile["user_rating"])
    if total_correct_tracked > 0:
        user_profile["last_session_accuracy"] = round(correct_count / total_correct_tracked, 3)

    save_json("output/rated_words.json", rated_words)
    save_json("output/user_profile.json", user_profile)

    print(f"[AI4] 세션 처리 완료. 새 userRating: {user_profile['user_rating']}")
    return {"new_user_rating": user_profile["user_rating"], "k_factor": k}


# ── 메인 ────────────────────────────────────────────────────────────────────

def main(args) -> None:
    if args.submit_result:
        process_session_result(args.submit_result)
        return

    for path, label in [
        ("output/rated_words.json", "rated_words.json (AI2 먼저 실행)"),
        ("output/user_profile.json", "user_profile.json (AI1 먼저 실행)"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{label} 없음.")
            sys.exit(1)

    rated_words = load_json("output/rated_words.json")
    user_profile = load_json("output/user_profile.json")
    daily_limit = args.daily_limit
    today = date.today()

    if args.today_only or args.days == 1:
        schedule = build_daily_schedule(rated_words, user_profile, daily_limit, today)
        save_json("output/daily_schedule.json", schedule)
        stats = schedule["stats"]
        print(
            f"[AI4] 오늘 스케줄 생성: 신규={stats['new_count']}, "
            f"복습={stats['review_count']}, 보충={stats['supplement_count']}"
        )
    else:
        days = args.days
        logger.info(f"{days}일 플랜 미리보기 생성 (파일은 오늘 스케줄만 저장)")
        for d in range(days):
            target_date = today + timedelta(days=d)
            sched = build_daily_schedule(rated_words, user_profile, daily_limit, target_date)
            if d == 0:
                save_json("output/daily_schedule.json", sched)
            logger.info(
                f"  {target_date}: 신규={sched['stats']['new_count']}, "
                f"복습={sched['stats']['review_count']}"
            )
        print(f"[AI4] {days}일 플랜 완료. 오늘 스케줄 → output/daily_schedule.json")


# ── 단독 테스트 ─────────────────────────────────────────────────────────────

def run_test() -> None:
    print("[AI4 TEST] FSRS 단독 테스트")
    today = date.today()

    # 초기 카드
    card = init_fsrs_card(3, today)  # rating=3 (Good)
    print(f"  init(rating=3): stability={card['stability']}, state={card['state']}")
    print(f"  due_date: {card['due_date'][:10]}")

    # Retrievability 테스트
    R = retrievability(card["stability"], card["stability"])
    print(f"  retrievability(t=S) = {R:.3f} (기대: ~0.9)")

    # 복습 후 stability
    review_day = today + timedelta(days=int(card["stability"]))
    card2 = process_review(card, 3, review_day)
    print(f"  after recall(rating=3): stability={card2['stability']}, state={card2['state']}")

    # Again → relearning
    card3 = process_review(card, 1, review_day)
    print(f"  after forget(rating=1): stability={card3['stability']}, state={card3['state']}")

    # k_factor
    for n, exp in [(0, 100), (5, 100), (6, 50), (21, 20)]:
        assert get_k_factor(n) == exp, f"k_factor({n}) 오류"
    print("  get_k_factor 검증 완료 ✓")

    # next_interval
    interval = next_interval(3.1262)  # w[2] 기본값
    print(f"  next_interval(S=3.1262) = {interval}일")

    print("[AI4 TEST] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4: 일별 학습 스케줄러")
    parser.add_argument("--today-only", action="store_true", help="오늘 스케줄만 생성")
    parser.add_argument("--days", type=int, default=1, help="생성할 일수")
    parser.add_argument("--daily-limit", type=int, default=100, help="하루 최대 단어 수")
    parser.add_argument("--submit-result", metavar="PATH", help="세션 결과 JSON 경로")
    parser.add_argument("--optimize", action="store_true", help="FSRS weights 최적화")
    parser.add_argument("--test", action="store_true", help="단독 기능 테스트")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.optimize:
        new_weights = optimize_weights()
        save_json("output/optimized_weights.json", {"weights": new_weights})
        print(f"[AI4] 최적화 완료 → output/optimized_weights.json")
    else:
        main(args)
