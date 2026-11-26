# IssueRadar – AI-Powered GitHub Issue Intelligence System

IssueRadar is an AI-powered GitHub Issue Intelligence Platform that analyzes public GitHub repositories, automatically classifies issues into meaningful categories, and identifies potentially similar issues using machine learning and text similarity techniques.

It helps developers understand repository issues, identify recurring problems, and explore issue patterns through an interactive dashboard.

## 🚀 Live Demo

- **Frontend:** https://issueradar-kkbf.onrender.com
- **Backend API:** https://issueradar-backend.onrender.com
- **API Health Check:** https://issueradar-backend.onrender.com/health

## ✨ Key Features

- **GitHub Repository Analysis:** Analyze public GitHub repositories using their repository URLs.
- **Automated Issue Classification:** Classify GitHub issues into five categories:
  - Bug
  - Feature Request
  - Documentation
  - Performance
  - Question
- **Similar Issue Detection:** Identify potentially similar issues using TF-IDF vectorization and cosine similarity.
- **Confidence Scores:** Display prediction confidence for classified issues.
- **Repository Analytics:** Visualize issue category distribution and repository statistics.
- **Interactive Dashboard:** View classified issues and their similar issue candidates in one place.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | React.js, Vite, CSS |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Text Processing | TF-IDF, Cosine Similarity |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Database / Search | CSV-based issue metadata and TF-IDF matrix |
| Deployment | Render |
| Version Control | Git, GitHub |

## 🧠 Machine Learning Approach

### 1. Issue Classification

IssueRadar uses a trained machine learning classifier to categorize GitHub issues into five predefined categories.

- Text preprocessing and feature extraction using TF-IDF.
- Classification using a trained Logistic Regression model.
- Prediction output includes the predicted category and confidence score.

### 2. Similar Issue Detection

The system identifies potentially similar issues using:

- **TF-IDF Vectorization:** Converts issue text into numerical feature vectors.
- **Cosine Similarity:** Measures similarity between issue vectors.
- **Repository Filtering:** Prioritizes matching issues from the same repository.
- **Similarity Threshold:** Filters results based on a configurable similarity score.

Note: Similarity indicates textual relevance and does not guarantee that two issues are actual duplicates.

## 📊 Dataset

The project uses a GitHub issue dataset collected from public repositories.

Repositories include:

- fastapi/fastapi
- pallets/flask
- django/django
- scikit-learn/scikit-learn
- pandas-dev/pandas
- microsoft/vscode

The dataset contains issue information such as titles, descriptions, categories, repository names, and issue URLs.

The classification dataset uses five categories: bug, feature_request, documentation, performance, and question.

## 🏗️ Project Structure

```text
IssueRadar/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── ml/
│   │       ├── duplicate_detector.py
│   │       └── ...
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
│
├── data/
├── docs/
├── models/
│   ├── issue_classifier.joblib
│   └── issue_faiss_metadata.csv
│
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10+
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/lakshmiprajapati/IssueRadar.git
cd IssueRadar
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment.

**Windows:**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Backend runs at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### 3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create a `.env` file inside the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## 🔌 API Endpoint

### Analyze Repository

**Endpoint:**

```http
POST /api/v1/analyze
```

**Request Body:**

```json
{
  "repository_url": "https://github.com/fastapi/fastapi"
}
```

The endpoint analyzes the repository and returns issue classification results, prediction confidence, repository analytics, and similar issue candidates.

### Health Check

```http
GET /health
```

Checks backend availability and classifier model status.

## 📈 Dashboard

The IssueRadar dashboard provides:

- Repository information and statistics.
- Issue category distribution.
- Total issues fetched and classified.
- Average prediction confidence.
- Classified issue details.
- Similar issue candidates.

## ☁️ Deployment

IssueRadar is deployed using Render.

- React frontend is deployed as a static site.
- FastAPI backend is deployed as a web service.
- Frontend communicates with the backend through an environment-configured API URL.

## 🔮 Future Enhancements

- Improve classification performance through model evaluation and tuning.
- Improve similar issue detection and reduce false positives.
- Fetch and analyze more issues using GitHub API pagination.
- Add advanced filtering and search options.
- Introduce issue trend analysis and historical insights.
- Explore semantic search techniques when deployment resources allow.

## 👩‍💻 Author

**Lakshmi Prajapati**

B.Tech – Information Technology

GitHub: https://github.com/lakshmiprajapati

---

⭐ If you find this project useful, consider giving it a star!
