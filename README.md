# Memorix Vocab Backend

FastAPI 기반 영어 단어 학습 백엔드. Oxford 3000/5000 DB, FSRS 스케줄링, IRT 유저 레이팅, CAT 온보딩 파이프라인 포함.

## 배포 정보

- **서버 URL**: `https://memorix-api-production.up.railway.app`
- **플랫폼**: Railway (Nixpacks 빌드)
- **DB**: PostgreSQL (Railway Managed)
- **Oxford 단어**: 4,968개 사전 적재 완료

### Unity 연동

Unity 클라이언트의 `serverUrl`을 아래로 변경:

```
# 로컬 개발
http://localhost:8000

# 프로덕션
https://memorix-api-production.up.railway.app
```

앱 최초 실행 시 `POST /api/users/create`로 `user_id`를 발급받아 `PlayerPrefs`에 저장하고, 이후 모든 API 호출에 포함시킵니다.

## 로컬 실행

### 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env
# .env에서 DATABASE_URL, ANTHROPIC_API_KEY 값 설정
```

### 서버 실행

```bash
uvicorn unity_bridge.server:app --host 0.0.0.0 --port 8000
```

서버 시작 시 `models/refined_db.json`에서 oxford_words 자동 시딩 (비어있을 때만).

## API 엔드포인트

| Method | Endpoint | 기능 |
|--------|----------|------|
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/users/create` | 유저 생성 → user_id 발급 |
| POST | `/api/upload-csv` | 단어 CSV 업로드 |
| GET | `/api/onboarding/quiz` | CAT 온보딩 퀴즈 단어 반환 |
| POST | `/api/onboarding/submit` | 퀴즈 결과 제출 → userRating 초기화 |
| GET | `/api/schedule/today` | 오늘의 학습 스케줄 |
| POST | `/api/session/result` | 학습 결과 제출 → FSRS + userRating 업데이트 |
| GET | `/api/user/profile` | 유저 프로필 반환 |

## 환경 변수

| 변수 | 설명 |
|------|------|
| `DATABASE_URL` | PostgreSQL 연결 URL |
| `ANTHROPIC_API_KEY` | Claude API 키 (단어 레이팅 검증용) |
