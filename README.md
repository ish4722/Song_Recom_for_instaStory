# 🎵 Image → Music Recommender

A multimodal recommendation system that analyzes an image, converts its visual context into semantic representations, retrieves mood-aligned songs with vector search, and optionally generates an original song from the same vibe.

## Architecture

```text
                         OFFLINE
Song metadata → descriptions → Sentence/CLIP embeddings → FAISS indexes
                                                        │
                         ONLINE                         │
Image → BLIP → Gemini → text embedding ────────────────┤
  │                                                     │
  └──────────────→ CLIP image embedding ────────────────┘
                              ↓
                    multimodal candidate retrieval
                              ↓
                 mood + preference-aware re-ranking
                              ↓
                    top-K recommendations
                              ↓
                         user feedback
                              ↓
                       personalization

Optional: image context → Gemini lyrics → Suno → generated audio
```

## Key Features

- **Visual understanding:** BLIP image captioning with optional Gemini refinement.
- **Multimodal retrieval:** combines semantic text retrieval with CLIP image-to-song retrieval.
- **Vector search:** FAISS indexes replace full-catalog ranking when indexes are available.
- **Preference-aware ranking:** mood and previous like/dislike feedback influence final ranking.
- **Explainable recommendations:** each result exposes semantic, visual and mood signals plus a human-readable reason.
- **Asynchronous generation:** Suno generation runs as a background job and is exposed through a status endpoint.
- **Offline preprocessing:** song embeddings and vector indexes are built once and reused at inference time.
- **Evaluation framework:** Precision@K, Recall@K, NDCG@K and MRR utilities are included without fabricating results.
- **Tests:** metric and ranking unit tests are included.

## Technology Stack

| Layer | Technology |
|---|---|
| Web/API | Flask |
| Vision | BLIP, OpenAI CLIP |
| LLM | Google Gemini |
| Text embeddings | Sentence Transformers (`all-mpnet-base-v2`) |
| Vector retrieval | FAISS |
| Audio generation | Suno + Playwright |
| Data | Pandas, Pickle |
| Testing | Pytest |

## Project Structure

```text
.
├── app.py
├── config.py
├── description.py
├── model.ipynb
├── requirements.txt
├── song_data.csv
├── song_data.pkl
├── .env.example
├── retrieval/
│   ├── faiss_index.py
│   └── ranker.py
├── services/
│   ├── feedback.py
│   └── model_manager.py
├── scripts/
│   └── build_indexes.py
├── evaluation/
│   ├── evaluate.py
│   └── evaluation_dataset.csv
├── tests/
│   └── test_recommender.py
├── templates/
│   ├── index.html
│   └── index_v2.html
├── suno_automation.py
└── suno_session_manager.py
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ish4722/Song_Recom_for_instaStory.git
cd Song_Recom_for_instaStory
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
```

### 2. Configure secrets

Copy `.env.example` to `.env` and add your own Gemini API key:

```text
GEMINI_API_KEY=your_key_here
FLASK_SECRET_KEY=your_random_secret
```

**Never commit `.env` or `suno_session.json`.** Any API key that was previously exposed in source control should be revoked and regenerated.

### 3. Build multimodal indexes

The repository contains the original `song_data.pkl`. To add CLIP embeddings and FAISS indexes:

```bash
python scripts/build_indexes.py
```

This creates:

- `song_data_multimodal.pkl`
- `song_text_faiss.index`
- `song_clip_faiss.index`

These generated artifacts are intentionally not required for the application to start: if they are absent, the app falls back to the original semantic retrieval path.

### 4. Run

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Recommendation Pipeline

### Candidate retrieval

Two independent representations are used:

1. **Text path:** image → BLIP/Gemini description → Sentence Transformer embedding → FAISS text index.
2. **Visual path:** image → CLIP image embedding → FAISS CLIP text index built from song descriptions.

The union of candidates is then re-ranked.

### Re-ranking

The final score combines semantic similarity, visual similarity, mood compatibility and user preference signals. Weights are configurable through environment variables instead of being hidden in application code.

## Evaluation

The evaluation module provides:

- Precision@K
- Recall@K
- NDCG@K
- MRR

The current `evaluation/evaluation_dataset.csv` is deliberately a **placeholder** because no labeled evaluation set has been supplied yet.

Expected schema:

```csv
image,expected_song_ids
image_001.jpg,Taylor Swift::Cruel Summer
image_002.jpg,"The Weeknd::Blinding Lights,Lana Del Rey::Summertime Sadness"
```

Populate the file with human relevance labels before running evaluation:

```bash
python evaluation/evaluate.py
```

Do not report evaluation numbers on the resume until the dataset has been populated and the experiments have actually been run.

## Personalization

The UI provides like/dislike feedback for recommended tracks. Feedback is stored per browser-generated user ID and used to adjust future ranking through artist preference signals.

This is intentionally a lightweight content-personalization layer. A future collaborative-filtering experiment can use public listening-interaction datasets such as the Million Song Dataset or Last.fm data once enough user interactions are available.

## AI Song Generation

The recommendation description can be passed to Gemini for short lyrics generation and then to the existing Suno/Playwright workflow. Generation is now exposed as a background job:

```text
POST /generate_song → job_id
GET /song_status/<job_id> → queued / completed / failed
```

Suno authentication may still require a one-time manual browser login/security check.

## Testing

```bash
pytest -q
```

## Limitations

- The recommendation quality is currently content-based; there is no trained collaborative filtering model.
- CLIP/FAISS indexes must be rebuilt when the song catalog changes.
- Suno browser automation depends on the external site's current UI and authentication flow.
- Quantitative recommendation results are intentionally pending the labeled evaluation dataset.

## Roadmap

- Populate the evaluation benchmark and run ablation studies: text-only vs CLIP-only vs multimodal fusion.
- Learn re-ranking weights from relevance labels instead of relying on initial configurable weights.
- Add richer user embeddings from interaction history.
- Expand the song catalog and metadata coverage.

## License

MIT License.
