"""
skillsync_app.py  v3.0
SkillSync backend — Flask REST API + ML engine + Email notifications + Google OAuth
New in v3: connect-request emails, Google OAuth verification, notification store
"""
import json, os, datetime, uuid, smtplib, threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL CONFIG  ← set env vars or fill in directly for testing
# ══════════════════════════════════════════════════════════════════════════════
EMAIL_CFG = {
    "enabled":      bool(os.environ.get("SMTP_EMAIL")),   # auto-enables if env set
    "smtp_host":    os.environ.get("SMTP_HOST",  "smtp.gmail.com"),
    "smtp_port":    int(os.environ.get("SMTP_PORT", "587")),
    "sender_email": os.environ.get("SMTP_EMAIL", ""),     # your gmail
    "sender_pass":  os.environ.get("SMTP_PASS",  ""),     # App Password
    "app_url":      os.environ.get("APP_URL",    "http://localhost:5050"),
}

# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY DATABASE
# ══════════════════════════════════════════════════════════════════════════════
USERS_DB    = {}   # id -> user dict
TEAMS_DB    = {}   # id -> team dict
CONNECT_DB  = {}   # request_id -> request dict
NOTIF_DB    = {}   # uid -> [notification dicts]

# ── Seed users ────────────────────────────────────────────────────────────────
SEED_USERS = [
    {"id":"arjun_iitm","username":"arjun.dev","name":"Arjun Sharma",
     "email":"arjun@demo.com","college":"IIT Madras","education":"4th Year",
     "github":"https://github.com/arjunsharma",
     "linkedin":"https://linkedin.com/in/arjunsharma","xp":3420,
     "domains":["ML / AI","Full-stack","Data Engineering"],
     "badges":["SIH Winner","5+ Hackathons"],
     "connects":["priya_nit","karthik_bits"],"teams":["team_demo1"],
     "date_joined":"2022-08-01","reviews":[]},
    {"id":"priya_nit","username":"priya.design","name":"Priya Nair",
     "email":"priya@demo.com","college":"NIT Trichy","education":"3rd Year",
     "github":"https://github.com/priyanair",
     "linkedin":"https://linkedin.com/in/priyanair","xp":2180,
     "domains":["Design / UX","Full-stack"],"badges":["Hack the Mountain"],
     "connects":["arjun_iitm"],"teams":["team_demo1"],
     "date_joined":"2023-01-15","reviews":[]},
    {"id":"karthik_bits","username":"karthik.hw","name":"Karthik Raj",
     "email":"karthik@demo.com","college":"BITS Pilani","education":"4th Year",
     "github":"https://github.com/karthikraj","linkedin":"","xp":4100,
     "domains":["Hardware / IoT","DevOps / Cloud"],
     "badges":["SIH Winner","Open Source Contributor"],
     "connects":["arjun_iitm"],"teams":["team_demo1"],
     "date_joined":"2021-06-10","reviews":[]},
    {"id":"sneha_iiith","username":"sneha.nlp","name":"Sneha Iyer",
     "email":"sneha@demo.com","college":"IIIT Hyderabad","education":"Postgraduate",
     "github":"https://github.com/sneha",
     "linkedin":"https://linkedin.com/in/sneha","xp":2890,
     "domains":["NLP / LLMs","ML / AI"],"badges":["AI Hack Finalist"],
     "connects":[],"teams":[],"date_joined":"2022-03-20","reviews":[]},
    {"id":"dev_vit","username":"dev.cloud","name":"Dev Patel",
     "email":"dev@demo.com","college":"VIT Vellore","education":"3rd Year",
     "github":"https://github.com/devpatel","linkedin":"","xp":1650,
     "domains":["DevOps / Cloud","Full-stack"],"badges":[],
     "connects":[],"teams":[],"date_joined":"2023-07-01","reviews":[]},
    {"id":"riya_nsut","username":"riya.web3","name":"Riya Mehta",
     "email":"riya@demo.com","college":"NSUT Delhi","education":"3rd Year",
     "github":"https://github.com/riyamehta",
     "linkedin":"https://linkedin.com/in/riyamehta","xp":2560,
     "domains":["Blockchain","Full-stack"],"badges":["ETHIndia Finalist"],
     "connects":[],"teams":[],"date_joined":"2022-11-05","reviews":[]},
    {"id":"aarav_dtu","username":"aarav.mobile","name":"Aarav Singh",
     "email":"aarav@demo.com","college":"DTU Delhi","education":"4th Year",
     "github":"https://github.com/aaravsingh",
     "linkedin":"https://linkedin.com/in/aaravsingh","xp":5200,
     "domains":["Mobile","Full-stack"],
     "badges":["Google DevFest Top 3","HackCBS Winner","5+ Hackathons"],
     "connects":[],"teams":[],"date_joined":"2021-01-20","reviews":[]},
]
for u in SEED_USERS:
    USERS_DB[u["id"]] = u
    NOTIF_DB[u["id"]] = []

SEED_TEAMS = [
    {"id":"team_demo1","name":"Team Firewall","hackathon":"Smart India Hackathon 2024",
     "start_date":"2024-01-24","end_date":"2024-01-26",
     "members":["arjun_iitm","karthik_bits","priya_nit"],
     "reviews":{
         "arjun_iitm":[{"from":"karthik_bits","from_name":"Karthik Raj","stars":5,
                         "text":"Brilliant ML work, led the team perfectly","date":"2024-01-27"}],
         "karthik_bits":[{"from":"arjun_iitm","from_name":"Arjun Sharma","stars":5,
                           "text":"Best hardware guy I've worked with, very reliable","date":"2024-01-27"}],
         "priya_nit":[{"from":"arjun_iitm","from_name":"Arjun Sharma","stars":4,
                        "text":"Great UI work under pressure","date":"2024-01-27"}],
     },"created_at":"2024-01-20"},
]
for t in SEED_TEAMS:
    TEAMS_DB[t["id"]] = t

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL HELPER
# ══════════════════════════════════════════════════════════════════════════════
def send_email_async(to_addr: str, subject: str, html_body: str):
    """Send email in background thread — never blocks the request."""
    if not EMAIL_CFG["enabled"] or not to_addr:
        print(f"[EMAIL SKIPPED] To={to_addr} | {subject}")
        return

    def _send():
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"SkillSync <{EMAIL_CFG['sender_email']}>"
            msg["To"]      = to_addr
            msg.attach(MIMEText(html_body, "html"))
            with smtplib.SMTP(EMAIL_CFG["smtp_host"], EMAIL_CFG["smtp_port"]) as server:
                server.ehlo()
                server.starttls()
                server.login(EMAIL_CFG["sender_email"], EMAIL_CFG["sender_pass"])
                server.sendmail(EMAIL_CFG["sender_email"], to_addr, msg.as_string())
            print(f"[EMAIL SENT] To={to_addr} | {subject}")
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")

    threading.Thread(target=_send, daemon=True).start()


def connect_request_email(from_user: dict, to_user: dict, req_id: str) -> str:
    """Build the HTML for a connection request email."""
    app_url = EMAIL_CFG["app_url"]
    accept_url  = f"{app_url}/api/connect/respond?req={req_id}&action=accept"
    decline_url = f"{app_url}/api/connect/respond?req={req_id}&action=decline"
    return f"""
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8">
    <style>
      body{{font-family:system-ui,sans-serif;background:#0a0a0b;color:#f0eff4;margin:0;padding:0}}
      .wrap{{max-width:520px;margin:0 auto;padding:40px 24px}}
      .logo{{font-size:22px;font-weight:900;letter-spacing:-1px;margin-bottom:28px}}
      .logo span{{color:#7c6ef6}}
      .card{{background:#111114;border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:28px}}
      h2{{font-size:20px;font-weight:800;margin-bottom:8px}}
      p{{font-size:14px;color:#9896a4;line-height:1.6;margin-bottom:20px}}
      .btn{{display:inline-block;padding:12px 28px;border-radius:8px;font-weight:700;
            font-size:14px;text-decoration:none;margin-right:10px}}
      .accept{{background:#7c6ef6;color:#fff}}
      .decline{{background:transparent;border:1px solid rgba(255,255,255,.15);color:#9896a4}}
      .avatar{{width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#7c6ef6,#a855f7);
               display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;
               color:#fff;margin-bottom:16px}}
      .footer{{font-size:12px;color:#5a5866;margin-top:24px;text-align:center}}
    </style></head>
    <body><div class="wrap">
      <div class="logo">skill<span>sync</span></div>
      <div class="card">
        <div class="avatar">{(from_user.get('username','?')[:2]).upper()}</div>
        <h2>You have a new connection request!</h2>
        <p><strong style="color:#f0eff4">{from_user.get('username','Someone')}</strong>
        ({from_user.get('college','')}) wants to connect with you on SkillSync.<br>
        They're working in: {', '.join(from_user.get('domains', []) or ['—'])}</p>
        <a href="{accept_url}"  class="btn accept">✓ Accept</a>
        <a href="{decline_url}" class="btn decline">✗ Decline</a>
      </div>
      <div class="footer">SkillSync · Sync your skills, build your dream team</div>
    </div></body></html>
    """


def add_notification(uid: str, notif: dict):
    """Push a notification to a user's inbox."""
    NOTIF_DB.setdefault(uid, []).insert(0, {
        **notif,
        "id":   str(uuid.uuid4())[:8],
        "ts":   datetime.datetime.utcnow().isoformat(),
        "read": False,
    })
    # Keep latest 50
    NOTIF_DB[uid] = NOTIF_DB[uid][:50]


# ══════════════════════════════════════════════════════════════════════════════
# ML ENGINE  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════
SKILL_CLUSTERS = {
    "ml_ai":      ["python","tensorflow","pytorch","keras","mlops","hugging face",
                   "nlp","computer vision","transformers","langchain","llm","rag","scikit"],
    "frontend":   ["react","vue","angular","typescript","javascript","css","figma",
                   "tailwind","next.js","redux","graphql","html"],
    "backend":    ["node.js","fastapi","django","flask","go","rust","java","express",
                   "rest api","grpc","postgresql","mongodb","redis"],
    "devops":     ["docker","kubernetes","aws","gcp","azure","terraform","ci/cd",
                   "linux","bash","nginx","prometheus"],
    "mobile":     ["react native","flutter","swift","kotlin","ios","android","firebase"],
    "hardware":   ["fpga","verilog","arduino","pcb design","embedded c","c++","rtos",
                   "raspberry pi","stm32","iot"],
    "blockchain": ["solidity","web3","ethereum","smart contracts","rust","polygon","hardhat"],
    "data":       ["pandas","numpy","sql","spark","kafka","airflow","dbt","tableau","etl"],
    "design":     ["figma","adobe xd","sketch","ui/ux","prototyping","typography",
                   "design systems","framer"],
    "cybersec":   ["penetration testing","burp suite","wireshark","ctf","cryptography",
                   "kali linux","owasp"],
}
DOMAIN_TO_CLUSTER = {
    "ML / AI":"ml_ai","Full-stack":"frontend","Mobile":"mobile",
    "Hardware / IoT":"hardware","Data Engineering":"data","DevOps / Cloud":"devops",
    "Design / UX":"design","Blockchain":"blockchain","Cybersecurity":"cybersec",
    "NLP / LLMs":"ml_ai",
}
XP_TIERS  = [0, 500, 1200, 2500, 4500, 6000]
XP_LABELS = ["Newcomer","Builder","Craftsperson","Architect","Legend"]


class SquadupEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="word", lowercase=True,
            token_pattern=r"[a-zA-Z0-9][a-zA-Z0-9/\.\+\-\_# ]*",
            max_features=500, ngram_range=(1,2),
        )
        self.fitted = False
        self._cache = {}

    def _to_text(self, user):
        parts = []
        for d in user.get("domains",[]):
            cl = DOMAIN_TO_CLUSTER.get(d,"")
            if cl and cl in SKILL_CLUSTERS:
                parts += SKILL_CLUSTERS[cl][:5]
            parts += [d.lower().replace("/"," ").replace("-"," ")] * 3
        for b in user.get("badges",[]): parts.append(b.lower())
        if user.get("college"):   parts.append(user["college"].lower())
        if user.get("education"): parts.append(user["education"].lower())
        return " ".join(parts) if parts else "hacker developer"

    def fit(self, users):
        texts = [self._to_text(u) for u in users]
        if not texts: return
        self.vectorizer.fit(texts)
        self.fitted = True
        self._cache = {u["id"]: self._vec(u) for u in users if u.get("id")}

    def _vec(self, user):
        if not self.fitted: return np.zeros(1)
        return self.vectorizer.transform([self._to_text(user)]).toarray()[0]

    def similarity(self, u1, u2):
        if not self.fitted: return 0.5
        v1 = self._cache.get(u1.get("id")) or self._vec(u1)
        v2 = self._cache.get(u2.get("id")) or self._vec(u2)
        raw = float(cosine_similarity([v1],[v2])[0][0])
        d1  = set(u1.get("domains",[])); d2 = set(u2.get("domains",[]))
        shared = len(d1 & d2); distinct = len(d1|d2) - shared
        xp_diff = abs(u1.get("xp",0) - u2.get("xp",0))
        final = raw*0.5 + max(0, shared*0.08 - distinct*0.03)*0.3 + max(0,1-xp_diff/5000)*0.2
        return min(1.0, max(0.0, final))

    def pairwise_compat(self, u1, u2):
        v1 = self._cache.get(u1.get("id")) or self._vec(u1)
        v2 = self._cache.get(u2.get("id")) or self._vec(u2)
        raw  = float(cosine_similarity([v1],[v2])[0][0]) if self.fitted else 0.5
        comp = max(0.0, min(1.0, 1.0 - abs(raw-0.25)*2.5))
        d1   = set(u1.get("domains",[])); d2 = set(u2.get("domains",[]))
        dom  = min(0.3, len(d1&d2)*0.12)
        lv_s = max(0, 1 - abs(u1.get("xp",0)-u2.get("xp",0))/5000)
        raw_score = comp*0.55 + dom*0.25 + lv_s*0.20
        return min(98, max(40, int(40 + raw_score*60)))

    def team_compat(self, members):
        if len(members) < 2: return {"score":0,"recommendation":"Add more members.","gaps":[]}
        all_domains = []
        for m in members: all_domains.extend(m.get("domains",[]))
        cluster_set = set()
        for d in all_domains:
            cl = DOMAIN_TO_CLUSTER.get(d)
            if cl: cluster_set.add(cl)
        coverage = int(min(100, len(cluster_set)*12))
        pairs = [(i,j) for i in range(len(members)) for j in range(i+1,len(members))]
        avg_compat = (sum(self.pairwise_compat(members[i],members[j]) for i,j in pairs)
                      / len(pairs)) if pairs else 70
        size_bonus  = {2:0,3:8,4:10,5:7}.get(len(members),5)
        final_score = min(98, int(avg_compat*0.7 + coverage*0.2 + size_bonus))
        all_cls = set(SKILL_CLUSTERS.keys())
        gaps = [c.replace("_"," ").title() for c in all_cls-cluster_set][:3]
        rec  = ("Excellent coverage! Add more members for even better coverage."
                if len(cluster_set)>=4 else f"Missing: {', '.join(gaps)}. Consider adding someone with those skills.")
        return {"score":final_score,"recommendation":rec,"covered":list(cluster_set),"gaps":gaps}

    def compose_team(self, anchor, pool, size=4):
        team      = [anchor]
        remaining = [u for u in pool if u.get("id") != anchor.get("id")]
        while len(team) < size and remaining:
            best, best_s = None, -1
            for c in remaining:
                s = self.team_compat(team+[c])["score"]
                if s > best_s: best_s=s; best=c
            if best: team.append(best); remaining.remove(best)
            else: break
        return team

    def suggestions(self, user, pool, n=8):
        others = [u for u in pool if u.get("id") != user.get("id")]
        scored = sorted(
            [{**u,"similarity_score":int(self.similarity(user,u)*100)} for u in others],
            key=lambda x: x["similarity_score"], reverse=True
        )
        return scored[:n]


engine = SquadupEngine()
engine.fit(list(USERS_DB.values()))
def refit(): engine.fit(list(USERS_DB.values()))

# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, origins=["*"])

def ok(data):   return jsonify({"success":True,"data":data})
def err(msg,c=400): return jsonify({"success":False,"error":msg}),c

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "skillsync_frontend.html")

@app.route("/")
def index():
    if os.path.exists(FRONTEND_PATH):
        return Response(open(FRONTEND_PATH,encoding="utf-8").read(), mimetype="text/html")
    return "<h2>Place skillsync_frontend.html next to this file</h2>"

@app.route("/api/health")
def health():
    return ok({"status":"ok","users":len(USERS_DB),"teams":len(TEAMS_DB),
                "requests":len(CONNECT_DB),"version":"skillsync-3.0"})

# ── Users ─────────────────────────────────────────────────────────────────────
@app.route("/api/users", methods=["GET","POST"])
def users_route():
    if request.method == "POST":
        p = request.json or {}
        if not p.get("id"): return err("Missing id")
        USERS_DB[p["id"]] = p
        NOTIF_DB.setdefault(p["id"], [])
        refit()
        return ok(p)
    q      = (request.args.get("q","")).lower()
    domain = request.args.get("domain","")
    users  = list(USERS_DB.values())
    if q:
        users = [u for u in users if
                 q in (u.get("username","")).lower() or
                 q in (u.get("name","")).lower() or
                 q in (u.get("college","")).lower() or
                 any(q in d.lower() for d in u.get("domains",[]))]
    if domain:
        users = [u for u in users if domain in u.get("domains",[])]
    return ok(users)

@app.route("/api/users/<uid>", methods=["GET","PATCH","DELETE"])
def user_route(uid):
    if request.method == "GET":
        u = USERS_DB.get(uid)
        return (ok(u) if u else err("Not found",404))
    if request.method == "PATCH":
        if uid not in USERS_DB: return err("Not found",404)
        USERS_DB[uid].update(request.json or {})
        refit()
        return ok(USERS_DB[uid])
    if request.method == "DELETE":
        USERS_DB.pop(uid,None)
        return ok({"deleted":uid})

@app.route("/api/users/<uid>/suggestions")
def user_suggestions(uid):
    u = USERS_DB.get(uid)
    if not u: return err("Not found",404)
    return ok(engine.suggestions(u, list(USERS_DB.values())))

# ── Connect requests ─────────────────────────────────────────────────────────
@app.route("/api/connect/request", methods=["POST"])
def connect_request():
    """
    Send a connection request + email notification to recipient.
    Body: { "from": uid, "to": uid }
    """
    body     = request.json or {}
    from_uid = body.get("from")
    to_uid   = body.get("to")

    from_user = USERS_DB.get(from_uid)
    to_user   = USERS_DB.get(to_uid)
    if not from_user: return err("Sender not found")
    if not to_user:   return err("Recipient not found")
    if to_uid in (from_user.get("connects") or []):
        return err("Already connected")

    # Check if pending request already exists
    existing = next((r for r in CONNECT_DB.values()
                     if r["from"]==from_uid and r["to"]==to_uid and r["status"]=="pending"), None)
    if existing:
        return ok({"request_id": existing["id"], "status":"already_pending"})

    req_id = str(uuid.uuid4())[:12]
    CONNECT_DB[req_id] = {
        "id":        req_id,
        "from":      from_uid,
        "from_name": from_user.get("username") or from_user.get("name","?"),
        "to":        to_uid,
        "to_name":   to_user.get("username") or to_user.get("name","?"),
        "status":    "pending",
        "created":   datetime.datetime.utcnow().isoformat(),
    }

    # In-app notification for recipient
    add_notification(to_uid, {
        "type":    "connect_request",
        "req_id":  req_id,
        "from_uid": from_uid,
        "from_name": from_user.get("username","?"),
        "from_college": from_user.get("college",""),
        "message": f"{from_user.get('username','?')} wants to connect with you",
    })

    # Email notification
    html = connect_request_email(from_user, to_user, req_id)
    send_email_async(to_user.get("email",""), "New connection request on SkillSync", html)

    return ok({"request_id": req_id, "status":"sent",
               "email_enabled": EMAIL_CFG["enabled"]})


@app.route("/api/connect/respond", methods=["POST","GET"])
def connect_respond():
    """
    Accept or decline a connection request.
    POST body: { "req_id": str, "action": "accept"|"decline" }
    GET params: ?req=<id>&action=<accept|decline>  (email link click)
    """
    if request.method == "GET":
        req_id = request.args.get("req")
        action = request.args.get("action","decline")
    else:
        body   = request.json or {}
        req_id = body.get("req_id")
        action = body.get("action","decline")

    req = CONNECT_DB.get(req_id)
    if not req: return err("Request not found",404)
    if req["status"] != "pending":
        if request.method == "GET":
            return Response(f"<h3>Request already {req['status']}</h3>", mimetype="text/html")
        return err(f"Request already {req['status']}")

    from_uid = req["from"]; to_uid = req["to"]
    from_user = USERS_DB.get(from_uid)
    to_user   = USERS_DB.get(to_uid)

    if action == "accept" and from_user and to_user:
        # Mutual connect
        from_user.setdefault("connects",[])
        to_user.setdefault("connects",[])
        if to_uid   not in from_user["connects"]: from_user["connects"].append(to_uid)
        if from_uid not in to_user["connects"]:   to_user["connects"].append(from_uid)
        # Give XP
        from_user["xp"] = from_user.get("xp",0) + 50
        to_user["xp"]   = to_user.get("xp",0) + 50
        refit()
        req["status"] = "accepted"
        # Notify sender
        add_notification(from_uid, {
            "type":    "connect_accepted",
            "from_uid": to_uid,
            "from_name": to_user.get("username","?"),
            "message": f"{to_user.get('username','?')} accepted your connection request! +50 XP",
        })
        # Email sender
        send_email_async(from_user.get("email",""),
                         f"🎉 {to_user.get('username','?')} accepted your request!",
                         f"<h2 style='font-family:system-ui'>Connected!</h2><p>{to_user.get('username','?')} accepted your SkillSync connection. +50 XP awarded.</p>")
        if request.method == "GET":
            return Response("<html><body style='font-family:system-ui;background:#0a0a0b;color:#f0eff4;display:flex;align-items:center;justify-content:center;min-height:100vh;'><div style='text-align:center'><h1>✓ Connected!</h1><p>You are now connected on SkillSync.</p><a href='/' style='color:#7c6ef6'>Return to SkillSync</a></div></body></html>", mimetype="text/html")
    else:
        req["status"] = "declined"
        if request.method == "GET":
            return Response("<html><body style='font-family:system-ui;background:#0a0a0b;color:#f0eff4;display:flex;align-items:center;justify-content:center;min-height:100vh;'><div style='text-align:center'><h2>Request declined.</h2><a href='/' style='color:#7c6ef6'>Return to SkillSync</a></div></body></html>", mimetype="text/html")

    return ok({"status": req["status"]})


@app.route("/api/connect/pending/<uid>")
def connect_pending(uid):
    """Get all pending incoming requests for a user."""
    pending = [r for r in CONNECT_DB.values() if r["to"]==uid and r["status"]=="pending"]
    return ok(pending)


# ── Notifications ─────────────────────────────────────────────────────────────
@app.route("/api/notifications/<uid>")
def get_notifications(uid):
    notifs = NOTIF_DB.get(uid, [])
    return ok(notifs)

@app.route("/api/notifications/<uid>/read", methods=["POST"])
def mark_read(uid):
    for n in NOTIF_DB.get(uid, []):
        n["read"] = True
    return ok({"ok":True})


# ── Google OAuth ──────────────────────────────────────────────────────────────
@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    """
    Verify Google ID token and return/create user.
    In production install: pip install google-auth
    and use: from google.oauth2 import id_token; id_token.verify_oauth2_token(...)
    For now: trusts the decoded payload sent from frontend for demo.
    """
    body  = request.json or {}
    ginfo = body.get("googleInfo",{})  # {email, name, picture, sub}
    email = ginfo.get("email","")
    if not email: return err("No email in Google payload")

    uid = "g_" + email.replace("@","_").replace(".","_")
    if uid in USERS_DB:
        return ok({"user": USERS_DB[uid], "is_new": False})

    # New Google user — create skeleton
    new_user = {
        "id":          uid,
        "username":    email.split("@")[0].replace(".","_"),
        "name":        ginfo.get("name","") or email.split("@")[0],
        "email":       email,
        "google_id":   ginfo.get("sub",""),
        "picture":     ginfo.get("picture",""),
        "college":     "",
        "education":   "1st Year",
        "github":      "",
        "linkedin":    "",
        "xp":          0,
        "domains":     [],
        "badges":      [],
        "connects":    [],
        "teams":       [],
        "date_joined": datetime.date.today().isoformat(),
        "reviews":     [],
        "_is_new":     True,
    }
    USERS_DB[uid] = new_user
    NOTIF_DB[uid] = []
    refit()
    return ok({"user": new_user, "is_new": True})


# ── Teams ─────────────────────────────────────────────────────────────────────
@app.route("/api/teams/compose", methods=["POST"])
def compose():
    body   = request.json or {}
    anchor = USERS_DB.get(body.get("anchor"))
    if not anchor: return err("Anchor not found")
    size = min(int(body.get("size",4)), len(USERS_DB))
    team = engine.compose_team(anchor, list(USERS_DB.values()), size)
    return ok({"team":team, "compatibility":engine.team_compat(team)})

@app.route("/api/compatibility/pair", methods=["POST"])
def pair_compat():
    body = request.json or {}
    u1 = USERS_DB.get(body.get("u1"))
    u2 = USERS_DB.get(body.get("u2"))
    if not u1 or not u2: return err("User(s) not found")
    return ok({"score":engine.pairwise_compat(u1,u2)})

@app.route("/api/teams", methods=["GET","POST"])
def teams_route():
    if request.method == "POST":
        t = request.json or {}
        if not t.get("id"): return err("Missing id")
        TEAMS_DB[t["id"]] = t
        # Notify all members
        for mid in t.get("members",[]):
            if mid != t.get("created_by"):
                add_notification(mid, {
                    "type": "team_added",
                    "team_name": t.get("name",""),
                    "message": f"You were added to team '{t.get('name','')}'",
                })
        return ok(t)
    uid   = request.args.get("user","")
    teams = [t for t in TEAMS_DB.values() if not uid or uid in t.get("members",[])]
    return ok(teams)

@app.route("/api/teams/<tid>", methods=["GET","PATCH"])
def team_route(tid):
    t = TEAMS_DB.get(tid)
    if not t: return err("Not found",404)
    if request.method == "PATCH":
        TEAMS_DB[tid].update(request.json or {})
        return ok(TEAMS_DB[tid])
    return ok(t)

@app.route("/api/teams/<tid>/reviews", methods=["POST"])
def add_review(tid):
    t = TEAMS_DB.get(tid)
    if not t: return err("Not found",404)
    body = request.json or {}
    to   = body.get("to")
    if not to: return err("Missing 'to'")
    rv = {"from":body.get("from",""),"from_name":body.get("from_name",""),
          "stars":int(body.get("stars",5)),"text":body.get("text",""),
          "date":datetime.date.today().isoformat()}
    if "reviews" not in t: t["reviews"] = {}
    t["reviews"].setdefault(to,[]).append(rv)
    target = USERS_DB.get(to)
    if target:
        bonus = {5:200,4:120,3:60}.get(rv["stars"],30)
        target["xp"] = target.get("xp",0) + bonus
        add_notification(to, {
            "type":    "review_received",
            "from_name": rv["from_name"],
            "stars":   rv["stars"],
            "message": f"{rv['from_name']} gave you a {rv['stars']}★ review! +{bonus} XP",
        })
    return ok(rv)

@app.route("/api/leaderboard")
def leaderboard():
    limit  = int(request.args.get("limit",20))
    scored = sorted(USERS_DB.values(), key=lambda u:u.get("xp",0), reverse=True)
    for i,u in enumerate(scored): u["rank"] = i+1
    return ok(scored[:limit])

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5050))
    print(f"""
  ╔══════════════════════════════════════╗
  ║  SkillSync v3.0  →  localhost:{port}  ║
  ║  Email: {'✓ ON' if EMAIL_CFG['enabled'] else '✗ off (set SMTP_EMAIL/SMTP_PASS)'}                   ║
  ╚══════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=port, debug=False)
