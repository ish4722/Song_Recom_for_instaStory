# 🎵 Image Song Recommender

<div align="center">
  <img src="static/uploads/Screenshot 2025-07-04 at 8.02.00 PM.png" width="600" alt="Image Song Recommender Interface">
  
  **Transform images into personalized music experiences with AI-powered song recommendations and custom audio generation**
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
</div>

---

## 🌟 Overview

Image Song Recommender is a multimodal Flask web application that bridges visual aesthetics with musical experiences. It analyzes uploaded images using vision-language models, retrieves songs that match the visual and semantic context, personalizes the ranking using user feedback, and can generate an original song from the same vibe.

The system has evolved from a semantic-only recommender into a **multimodal retrieval and ranking pipeline**, combining image understanding, text embeddings, CLIP visual embeddings, FAISS vector search, mood signals, and lightweight personalization.

### 🎯 Key Highlights
- **AI-Powered Image Analysis**: Combines BLIP image captioning with Google Gemini for contextual understanding
- **Multimodal Recommendations**: Combines semantic text similarity with CLIP-based visual similarity
- **Fast Vector Retrieval**: Uses FAISS indexes for efficient candidate retrieval
- **Personalized Ranking**: Incorporates mood compatibility and user like/dislike feedback
- **Explainable Results**: Shows why a recommended song matches the uploaded image
- **Custom Song Creation**: Generate unique lyrics and audio using Gemini + Suno.ai
- **Asynchronous Generation**: Song generation runs as a background job instead of blocking the recommendation request
- **Evaluation Ready**: Includes Precision@K, Recall@K, NDCG@K and MRR evaluation utilities
- **Model Caching**: Expensive BLIP, CLIP and Sentence Transformer models are loaded once and reused

---

## ✨ Features

### 🖼️ **Intelligent Image Analysis**
- Upload any image and receive an AI-generated description
- BLIP extracts the initial visual context
- Gemini optionally refines the description into a richer semantic representation
- Optional manual description input for additional context

### 🎵 **Multimodal Personalized Recommendations**
- Sentence Transformers generate semantic text embeddings
- CLIP maps image and song-description representations into a shared visual-semantic space
- FAISS retrieves candidates efficiently from both modalities
- Results are re-ranked using semantic similarity, visual similarity, mood compatibility and user preferences
- Language and artist filters remain available

### 💡 **Explainable Recommendations**
Each recommendation exposes individual ranking signals such as:
- Semantic similarity
- Visual similarity
- Mood compatibility
- Preference influence
- A human-readable explanation of the match

### 🎼 **Custom Song Generation**
- AI-generated lyrics tailored to the image's mood and context
- Audio creation through the existing Suno.ai + Playwright workflow
- Generation is handled asynchronously using a job ID and status polling
- Downloadable generated audio when the Suno workflow completes

### 🎨 **Modern Interface**
- Responsive web interface
- Image upload with contextual controls
- Mood, language and artist filtering
- Interactive recommendation cards
- Like/dislike feedback for personalization
- Real-time generation status updates

---

## 🏗️ System Architecture

The project now separates **offline indexing** from **online recommendation**, making the expensive embedding and indexing work reusable at inference time.

```text
                              OFFLINE PIPELINE
┌─────────────────────────────────────────────────────────────────────────┐
│ Song Metadata → Descriptions → Sentence Transformer Embeddings        │
│                              └──────→ CLIP Text Embeddings              │
│                                         ↓                               │
│                              FAISS Vector Indexes                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ precomputed
                                      ▼
                              ONLINE PIPELINE
┌─────────────────────────────────────────────────────────────────────────┐
│ Uploaded Image                                                         │
│      │                                                                  │
│      ├────────→ BLIP → Gemini → Text Embedding ───────┐                │
│      │                                                  │                │
│      └────────→ CLIP Image Embedding ──────────────────┤                │
│                                                         ▼                │
│                                           Multimodal Candidate Retrieval │
│                                                         ↓                │
│                                      Mood + Preference-aware Re-ranking │
│                                                         ↓                │
│                                           Top-K Recommendations         │
│                                                         ↓                │
│                                             User Feedback               │
│                                                         ↓                │
│                                              Personalization            │
└─────────────────────────────────────────────────────────────────────────┘

Optional Generation:
Image Context → Gemini Lyrics → Background Job → Suno.ai → Generated Audio
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask |
| **Image Processing** | BLIP (`Salesforce/blip-image-captioning-base`) |
| **Visual Embeddings** | OpenAI CLIP (`openai/clip-vit-base-patch32`) |
| **AI Generation** | Google Gemini API |
| **Text Embeddings** | Sentence Transformers (`all-mpnet-base-v2`) |
| **Vector Search** | FAISS |
| **Audio Creation** | Suno.ai |
| **Browser Automation** | Playwright |
| **Data Processing** | Pandas, Pickle |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Testing** | Pytest |

---

## 📁 Project Structure

```text
image-song-recommender/
├── 📁 static/
│   ├── 🎵 audio/                    # Generated audio files
│   └── 📷 uploads/                  # Temporary image storage
├── 📁 templates/
│   ├── 🌐 index.html                # Original web interface
│   └── 🌐 index_v2.html              # Updated recommendation interface
├── 📁 retrieval/
│   ├── 🔎 faiss_index.py            # FAISS index creation and search
│   └── 📊 ranker.py                 # Multisignal ranking + explanations
├── 📁 services/
│   ├── ❤️ feedback.py               # User feedback and preferences
│   └── 🧠 model_manager.py          # Cached model loading
├── 📁 scripts/
│   └── ⚙️ build_indexes.py           # Offline embedding/index builder
├── 📁 evaluation/
│   ├── 📈 evaluate.py               # Recommendation metrics
│   └── 🧪 evaluation_dataset.csv    # Placeholder for human relevance labels
├── 📁 tests/
│   └── ✅ test_recommender.py       # Unit tests
├── 🐍 app.py                        # Flask application and API routes
├── 🧠 description.py                # Image understanding + text embeddings
├── ⚙️ config.py                     # Environment/configuration settings
├── 📊 model.ipynb                   # Original data processing notebook
├── 📋 requirements.txt              # Dependencies
├── 📄 song_data.csv                 # Raw song metadata
├── 💾 song_data.pkl                 # Original song embeddings/data
├── 🤖 suno_automation.py            # Suno.ai automation
└── 🔧 suno_session_manager.py       # Suno.ai session management
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Internet connection for AI services
- Playwright browser binaries for Suno generation

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ish4722/Song_Recom_for_instaStory.git
   cd Song_Recom_for_instaStory
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install
   ```

4. **Configure environment variables**

   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

   Then add your own credentials:
   ```text
   GEMINI_API_KEY=your_key_here
   FLASK_SECRET_KEY=your_random_secret
   ```

   **Never commit `.env` or `suno_session.json`.** If an API key was previously exposed in source control, revoke it and generate a new one.

5. **Verify song data**

   Ensure `song_data.pkl` exists in the root directory. If it is missing, run the data-processing workflow in `model.ipynb` to generate it.

### ⚡ Build the Multimodal Indexes

The original `song_data.pkl` can still be used with the fallback semantic retrieval path. For the full multimodal pipeline, build the additional CLIP embeddings and FAISS indexes:

```bash
python scripts/build_indexes.py
```

This generates:
- `song_data_multimodal.pkl`
- `song_text_faiss.index`
- `song_clip_faiss.index`

These generated artifacts are ignored by Git and should be rebuilt whenever the song catalog changes.

### Running the Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` in your browser.

---

## 🎯 Usage Guide

### 1. Upload & Analyze
- **Upload an image** by clicking the file input or dragging and dropping
- **Add context** with an optional manual description
- **Select a mood** when you want to guide the ranking
- **Set preferences** using language and artist filters

### 2. Get Recommendations
- Click **Get Song Recommendations**
- View the AI-generated image description
- Explore ranked song suggestions
- Compare semantic, visual and mood signals
- Like or dislike recommendations to improve future ranking

### 3. Generate Custom Songs
- Click **Generate Song from This Description**
- A background job is created and a job ID is returned
- The interface polls the job status while Suno processes the request
- Complete any required Suno.ai authentication/security steps when prompted
- Play or download the generated audio when available

---

## 🧠 Recommendation Pipeline

### 1. Image Understanding

The uploaded image follows two complementary paths:

**Semantic path:**
```text
Image → BLIP → Gemini refinement → Sentence Transformer → Text embedding
```

**Visual path:**
```text
Image → CLIP image encoder → Visual embedding
```

### 2. Candidate Retrieval

The two embeddings retrieve candidates independently from FAISS indexes:

- **Text FAISS index** → semantically similar song descriptions
- **CLIP FAISS index** → visually compatible song descriptions

The candidate sets are combined before re-ranking.

### 3. Multisignal Re-ranking

The final ranking combines multiple signals:

```text
Final Score =
    Text Similarity × 0.48
  + Visual Similarity × 0.32
  + Mood Compatibility × 0.12
  + Preference Signal × 0.08
```

The weights are configurable through environment variables in `config.py`, making it possible to tune the recommender later using labeled evaluation data.

### 4. Personalization

User feedback is stored using a browser-generated user ID. Likes and dislikes contribute artist-level preference signals that influence subsequent recommendations.

This is currently a lightweight **content-personalization layer**. A stronger collaborative-filtering system could later use listening-interaction datasets such as the Million Song Dataset or Last.fm data once sufficient interaction history is available.

---

## 💡 Explainable Recommendations

Instead of returning only a similarity score, the recommender exposes the signals behind each result. Example explanations can include:

- Strong semantic match with the image description
- Strong visual-to-music embedding match
- Compatible mood cues
- Artist preference match based on previous feedback

This makes the ranking easier to inspect and debug while developing the recommendation system.

---

## 📊 Evaluation

The repository includes an evaluation framework with:

- **Precision@K**
- **Recall@K**
- **NDCG@K**
- **MRR**

The file `evaluation/evaluation_dataset.csv` is intentionally a **placeholder** because a labeled image-to-song evaluation dataset has not yet been provided.

### Expected Format

```csv
image,expected_song_ids
image_001.jpg,Taylor Swift::Cruel Summer
image_002.jpg,"The Weeknd::Blinding Lights,Lana Del Rey::Summertime Sadness"
```

Replace the placeholder with human relevance labels before running evaluation:

```bash
python evaluation/evaluate.py
```

No evaluation numbers are claimed in this README until the dataset is populated and the experiments are actually run.

A useful future experiment is an ablation comparison of:

```text
Text-only retrieval
        vs
CLIP-only retrieval
        vs
Multimodal fusion
```

---

## ⚙️ Configuration

### Gemini API Setup

The API key is now loaded through environment variables rather than being stored directly in the source code.

1. Copy `.env.example` to `.env`
2. Add your Gemini API key
3. Restart the application

### Ranking Configuration

The main ranking weights can be adjusted through environment variables:

```text
TEXT_WEIGHT=0.48
IMAGE_WEIGHT=0.32
MOOD_WEIGHT=0.12
PREFERENCE_WEIGHT=0.08
```

### Suno.ai Authentication
- First-time users may see an automated browser window
- Complete the Google login process manually when required
- Complete any security challenge shown by Suno.ai
- The session can then be reused through the session manager

---

## 🔄 Asynchronous Song Generation

Suno generation no longer needs to hold the recommendation request open while the external automation runs.

```text
POST /generate_song
        ↓
     job_id
        ↓
Background Suno Worker
        ↓
GET /song_status/<job_id>
        ↓
queued → completed / failed
```

This keeps the web application responsive while the longer-running generation process executes in the background.

---

## 🧪 Testing

Run the unit tests with:

```bash
pytest -q
```

The test suite covers recommendation metrics and core ranking/preference behavior.

---

## 🔮 Roadmap

- [ ] **Evaluation Benchmark** - Build a labeled image-to-song relevance dataset and run ablation studies
- [ ] **Learned Ranking** - Learn optimal ranking weights from relevance labels
- [ ] **Richer Personalization** - Build user embeddings from longer interaction histories
- [ ] **Video Recommendations** - Extend image understanding to short-form video content
- [ ] **Enhanced Mobile Experience** - Improve responsive design for mobile users
- [ ] **User Accounts** - Save preferences and recommendation history
- [ ] **Audio Previews** - Spotify API integration for track previews
- [ ] **Advanced Filters** - Genre, mood, era and other metadata-based filtering
- [ ] **Batch Processing** - Analyze multiple images at once
- [ ] **Social Features** - Share recommendations and custom songs

---

## ⚠️ Limitations

- Recommendation quality is currently primarily content-based; there is no trained collaborative-filtering model yet.
- CLIP and FAISS artifacts must be rebuilt when the song catalog changes.
- The evaluation dataset is currently a placeholder, so quantitative recommendation performance has not been established.
- Suno browser automation depends on the external site's current UI and authentication flow.
- Gemini and other AI services require their respective external APIs and may change over time.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[BLIP](https://github.com/salesforce/BLIP)** - Image captioning and visual understanding
- **[Google Gemini](https://ai.google.dev/)** - Description refinement and lyrics generation
- **[OpenAI CLIP](https://github.com/openai/CLIP)** - Multimodal image-text representation
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient vector similarity search
- **[Suno.ai](https://suno.ai/)** - AI-powered music generation platform
- **[Sentence Transformers](https://www.sbert.net/)** - Efficient semantic similarity matching
- **[Playwright](https://playwright.dev/)** - Browser automation

---

<div align="center">
  <p>Made with ❤️ by Ishan</p>
  <p>
    <a href="https://github.com/ish4722/Song_Recom_for_instaStory/issues">Report Bug</a>
    ·
    <a href="https://github.com/ish4722/Song_Recom_for_instaStory/issues">Request Feature</a>
  </p>
</div>
