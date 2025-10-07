# 🚀 Panduan Deploy ke Hosting Gratis

## 📋 Persiapan Sebelum Deploy

### 1. Setup Git & GitHub

Jika belum punya repository, buat dulu:

```bash
# Di folder project
git init
git add .
git commit -m "Initial commit"

# Buat repository di GitHub (https://github.com/new)
# Kemudian:
git remote add origin https://github.com/username/yt-sentiment.git
git branch -M main
git push -u origin main
```

### 2. File yang Sudah Disiapkan

✅ **render.yaml** - Config untuk Render.com
✅ **Procfile** - Config untuk web service
✅ **runtime.txt** - Python version
✅ **requirements.txt** - Dependencies (sudah diupdate)
✅ **.gitignore** - Exclude files yang tidak perlu
✅ **app.py** - Sudah production-ready

---

## 🎯 Opsi 1: Deploy ke Render.com (RECOMMENDED)

### Kelebihan:
- ✅ 100% Gratis selamanya
- ✅ Auto-deploy dari GitHub
- ✅ Free SSL certificate
- ✅ Setup mudah (5-10 menit)
- ✅ Support environment variables

### Limitasi:
- Sleep setelah 15 menit tidak ada traffic
- 750 jam/bulan (cukup untuk development)
- Cold start ~30 detik setelah sleep

---

### Langkah-langkah Deploy:

#### 1. Push Code ke GitHub

```bash
# Pastikan semua file sudah di-commit
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 2. Daftar di Render.com

1. Buka https://render.com
2. Sign up dengan GitHub account
3. Klik **"Connect GitHub"** dan authorize Render

#### 3. Create New Web Service

1. Di Dashboard, klik **"New +"** → **"Web Service"**
2. Pilih repository: **yt-sentiment** (atau nama repo kamu)
3. Klik **"Connect"**

#### 4. Configure Service

**Settings yang perlu diisi:**

| Field | Value |
|-------|-------|
| **Name** | `yt-sentiment-analysis` |
| **Region** | Singapore (terdekat dengan Indonesia) |
| **Branch** | `main` |
| **Root Directory** | (kosongkan) |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |

#### 5. Set Environment Variables

Di section **"Environment Variables"**, tambahkan:

```
YOUTUBE_API_KEY = <your_youtube_api_key>
SECRET_KEY = <generate_random_string>
PYTHON_VERSION = 3.11.0
```

**Cara generate SECRET_KEY:**
```python
import secrets
print(secrets.token_hex(32))
# Copy hasilnya
```

#### 6. Deploy!

1. Klik **"Create Web Service"**
2. Tunggu 5-10 menit (build & deploy)
3. Lihat logs untuk memastikan tidak ada error
4. Setelah selesai, URL akan muncul: `https://yt-sentiment-analysis.onrender.com`

#### 7. Test Aplikasi

Buka URL yang diberikan dan test:
- ✅ Homepage loading
- ✅ Scraping works
- ✅ Analysis works
- ✅ All features functional

---

## 🎯 Opsi 2: Deploy ke Railway.app

### Kelebihan:
- ✅ $5 kredit gratis/bulan
- ✅ Tidak sleep (always on)
- ✅ Faster cold start
- ✅ Better performance

### Limitasi:
- Perlu credit card (tapi tidak di-charge)
- $5 kredit = ~500 jam/bulan

---

### Langkah-langkah Deploy:

#### 1. Daftar di Railway.app

1. Buka https://railway.app
2. Sign up dengan GitHub
3. Verify dengan credit card (untuk $5 kredit gratis)

#### 2. Create New Project

1. Klik **"New Project"**
2. Pilih **"Deploy from GitHub repo"**
3. Pilih repository: **yt-sentiment**

#### 3. Configure Environment Variables

1. Klik project → **"Variables"** tab
2. Tambahkan:
   ```
   YOUTUBE_API_KEY = <your_api_key>
   SECRET_KEY = <random_string>
   PORT = 8080
   ```

#### 4. Deploy

1. Railway otomatis deploy
2. Tunggu 5-10 menit
3. Klik **"Settings"** → **"Generate Domain"**
4. URL akan muncul: `https://yt-sentiment.up.railway.app`

---

## 🎯 Opsi 3: Deploy ke PythonAnywhere

### Kelebihan:
- ✅ Khusus Python, tidak sleep
- ✅ Web console built-in
- ✅ Free tier bagus

### Limitasi:
- Domain: username.pythonanywhere.com
- Manual deployment
- 512MB disk space

---

### Langkah-langkah Deploy:

#### 1. Daftar di PythonAnywhere

1. Buka https://www.pythonanywhere.com
2. Sign up (Free tier)
3. Verify email

#### 2. Upload Code

**Cara 1: Git Clone**
```bash
# Di Bash console (PythonAnywhere)
git clone https://github.com/username/yt-sentiment.git
cd yt-sentiment
```

**Cara 2: Upload Files**
- Upload zip via Files tab
- Extract di directory

#### 3. Create Virtual Environment

```bash
# Di Bash console
cd /home/username/yt-sentiment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Setup Web App

1. Klik **"Web"** tab
2. Klik **"Add a new web app"**
3. Pilih **"Manual configuration"**
4. Pilih **"Python 3.11"**

#### 5. Configure WSGI

Edit file WSGI (di Web tab):

```python
import sys
import os

# Add project directory
project_home = '/home/username/yt-sentiment'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load environment variables
os.environ['YOUTUBE_API_KEY'] = 'your_api_key_here'
os.environ['SECRET_KEY'] = 'your_secret_key_here'

from app import app as application
```

#### 6. Set Virtual Environment

Di Web tab:
- **Virtualenv:** `/home/username/yt-sentiment/venv`

#### 7. Reload & Test

1. Klik **"Reload"** button
2. Buka: `https://username.pythonanywhere.com`

---

## 🔧 Troubleshooting

### 1. Build Failed - Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'xxx'
```

**Solution:**
- Pastikan `requirements.txt` lengkap
- Cek spelling dependencies
- Update requirements: `pip freeze > requirements.txt`

### 2. Application Timeout

**Error:**
```
Application failed to start within 60 seconds
```

**Solution:**
- Turunkan versi library yang berat (matplotlib, numpy)
- Hapus dependencies yang tidak dipakai (tensorflow, torch)
- Optimize startup time

### 3. Memory Exceeded

**Error:**
```
Memory limit exceeded (512MB)
```

**Solution:**
- Render: Upgrade ke paid plan ($7/month untuk 512MB)
- Railway: Sudah 1GB di free tier
- PythonAnywhere: Optimize memory usage

### 4. Environment Variables Not Working

**Solution:**
- Pastikan KEY di-set dengan benar
- Restart service setelah update env vars
- Check logs: `os.getenv('KEY')` return None

### 5. Cold Start Lambat (Render)

**Normal:** First request setelah sleep bisa 30-60 detik

**Workaround:**
- Setup cron job untuk ping app tiap 10 menit
- Upgrade ke paid plan (tidak sleep)
- Gunakan Railway (tidak sleep)

---

## 📊 Perbandingan Hosting

| Feature | Render | Railway | PythonAnywhere |
|---------|--------|---------|----------------|
| **Price** | Free | $5 kredit | Free |
| **Memory** | 512MB | 1GB | 512MB |
| **Sleep** | Ya (15 min) | Tidak | Tidak |
| **Domain** | Custom | Custom | username.pythonanywhere |
| **SSL** | ✅ Auto | ✅ Auto | ✅ Auto |
| **Setup** | 5 min | 5 min | 15 min |
| **Auto Deploy** | ✅ | ✅ | ❌ Manual |
| **Best For** | Side projects | Production | Learning |

---

## 🎯 Recommended: Render.com

**Pilih Render jika:**
- ✅ Side project / portfolio
- ✅ Traffic rendah-medium
- ✅ Mau setup cepat
- ✅ Tidak masalah dengan cold start

**Pilih Railway jika:**
- ✅ Need always-on
- ✅ Production app
- ✅ Punya credit card
- ✅ Need better performance

**Pilih PythonAnywhere jika:**
- ✅ Belajar deploy
- ✅ Tidak punya credit card
- ✅ Mau kontrol penuh
- ✅ Domain tidak masalah

---

## 🚀 After Deploy Checklist

### 1. Test Semua Fitur

- [ ] Homepage loading
- [ ] Scraping YouTube comments
- [ ] Sentiment analysis (4 methods)
- [ ] Quick analysis modal
- [ ] Preprocessing modal
- [ ] Word clouds generation
- [ ] Export to Excel
- [ ] Save/load analysis

### 2. Setup Monitoring

**Render:**
- Cek logs di Dashboard
- Setup notification untuk errors

**Railway:**
- Cek metrics di Dashboard
- Setup webhooks untuk alerts

### 3. Custom Domain (Optional)

**Render:**
1. Dashboard → Settings → Custom Domain
2. Add CNAME record di DNS provider
3. Format: `app.yourdomain.com` → `yt-sentiment.onrender.com`

**Railway:**
1. Settings → Domains → Add Custom Domain
2. Add CNAME: `app.yourdomain.com` → `yt-sentiment.up.railway.app`

### 4. Setup Cron Job (Untuk Render)

Agar tidak sleep, ping tiap 10 menit:

**Cron-job.org:**
1. Daftar di https://cron-job.org
2. Create new cron job
3. URL: `https://yt-sentiment.onrender.com`
4. Interval: Every 10 minutes
5. Activate

### 5. Analytics (Optional)

Add Google Analytics:

```html
<!-- Di templates/base.html, sebelum </head> -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

---

## 💡 Pro Tips

### 1. Optimize Loading Time

```python
# Di app.py, lazy load analyzer
@app.before_first_request
def initialize():
    global analyzer
    if analyzer is None:
        analyzer = SentimentAnalyzer()
```

### 2. Cache Results

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def analyze_cached(text):
    return analyzer.analyze(text)
```

### 3. Async Processing

Untuk analisis besar, gunakan background job:
- Render: Redis + Celery
- Railway: Built-in queues
- PythonAnywhere: Task queue

### 4. Database Integration

Jika mau save data persistent:
- Render: PostgreSQL free
- Railway: PostgreSQL/Redis free
- PythonAnywhere: MySQL free

### 5. Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    ...
```

---

## 📚 Resources

### Documentation:
- Render: https://render.com/docs
- Railway: https://docs.railway.app
- PythonAnywhere: https://help.pythonanywhere.com

### Tutorials:
- Deploy Flask to Render: https://render.com/docs/deploy-flask
- Deploy Flask to Railway: https://railway.app/template/flask
- Deploy Flask to PythonAnywhere: https://help.pythonanywhere.com/pages/Flask/

### Support:
- Render Community: https://community.render.com
- Railway Discord: https://discord.gg/railway
- PythonAnywhere Forum: https://www.pythonanywhere.com/forums/

---

## 🎉 Selamat!

Aplikasi kamu sudah online dan bisa diakses dari mana saja! 🚀

**Share URL:**
- Portfolio: Taruh di CV/LinkedIn
- GitHub: Add link di README
- Social Media: Share ke teman-teman

**Next Steps:**
- [ ] Add custom domain
- [ ] Setup monitoring
- [ ] Add more features
- [ ] Optimize performance
- [ ] Get user feedback

---

**Status:** ✅ Ready to Deploy  
**Estimated Time:** 10-15 minutes  
**Difficulty:** ⭐⭐ (Easy)

🎯 **Good luck dengan deployment!** 🎯
