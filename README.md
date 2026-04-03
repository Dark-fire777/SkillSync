# SkillSync 🚀

> **AI-powered hackathon team matching for Indian college students.**  
> Find teammates by skill complementarity, build squads, review collaborators — all in one platform.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/skillsync/blob/main/colab_launcher.ipynb)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.3%2B-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔗 Project Link

**GitHub Repository:** [https://github.com/Dark-fire777/skillsync](https://github.com/Dark-fire777/SkillSync)

> ⚠️ Replace `YOUR_USERNAME` with your actual GitHub username after pushing.

---

## ✨ Features

- **AI Team Composer** — TF-IDF + cosine similarity engine suggests teammates based on skill complementarity
- **Skill Domains** — 10+ domains (ML/AI, Full-stack, Design, Blockchain, Hardware/IoT, NLP, DevOps, Mobile, etc.)
- **XP & Leaderboard** — earn XP from hackathons, reviews, and connections
- **Post-hackathon Reviews** — star ratings and written feedback for teammates
- **Connections** — instant one-click connect with other hackers
- **Notifications** — real-time bell notifications for reviews and team activity
- **Email Alerts** — SMTP-based email notifications (optional, via Gmail App Password)
- **Google OAuth Ready** — optional Google sign-in verification hook
- **Colab Launcher** — run the full stack in a browser with zero local setup via ngrok

---

## 📁 Project Structure

```
skillsync/
├── app.py                  # Flask REST API + ML engine (v3.0)
├── frontend.html           # Single-file frontend (HTML/CSS/JS)
├── colab_launcher.ipynb    # Google Colab one-click launcher
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Option A — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/skillsync.git
cd skillsync

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the backend
python app.py
# → Server running at http://localhost:5050

# 4. Open frontend
# Open frontend.html in your browser directly,
# or serve it:
python -m http.server 8080
# → Visit http://localhost:8080/frontend.html
```

### Option B — Run on Google Colab (no setup needed)

Click the badge:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/skillsync/blob/main/colab_launcher.ipynb)

The notebook installs dependencies, starts the Flask server, and exposes a public URL via ngrok automatically.

---

## ⚙️ Environment Variables (Optional)

| Variable | Description | Default |
|---|---|---|
| `SMTP_EMAIL` | Gmail address for email notifications | *(disabled)* |
| `SMTP_PASS` | Gmail App Password | *(disabled)* |
| `SMTP_HOST` | SMTP host | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `APP_URL` | Public URL shown in emails | `http://localhost:5050` |
| `PORT` | Port to run the server on | `5050` |

Set them before running:
```bash
export SMTP_EMAIL=you@gmail.com
export SMTP_PASS=your_app_password
python app.py
```

---

## 🛠️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/users` | List all users |
| `POST` | `/api/users` | Register a new user |
| `GET/PATCH` | `/api/users/<id>` | Get or update a user |
| `GET` | `/api/users/<id>/suggestions` | AI-matched teammate suggestions |
| `POST` | `/api/teams` | Create a team |
| `GET/PATCH` | `/api/teams/<id>` | Get or update a team |
| `POST` | `/api/teams/<id>/reviews` | Submit a peer review |
| `POST` | `/api/teams/compose` | AI team composer |
| `GET` | `/api/leaderboard` | XP leaderboard |
| `GET` | `/api/notifications/<uid>` | Get notifications |
| `POST` | `/api/notifications/<uid>/read` | Mark all as read |

---

## 🧠 How the AI Matching Works

1. Each user's profile is encoded as a text document: skills, domains, tools, and bio keywords.
2. All profiles are vectorized using **TF-IDF**.
3. **Cosine similarity** is computed between the current user and all others.
4. Suggestions are ranked by complementarity score — prioritising skill gaps, not clones.
5. The team composer picks the optimal N-person squad to maximise collective domain coverage.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

*Built for Indian hackathon culture — from IITs to NITs to VITs. 🇮🇳*
