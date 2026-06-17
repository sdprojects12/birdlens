# 🦅 BirdLens

AI-powered bird identifier built with BioCLIP 2 + eBird API.

## Project Structure

```
BirdLens/
├── app.py              ← Flask backend (API routes)
├── requirements.txt    ← Python dependencies
├── static/
│   └── index.html      ← Full frontend (single file)
└── README.md
```

## Setup (do this once)

### 1. Create & activate a virtual environment

```bash
# In your BirdLens folder:
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ pybioclip will download the BioCLIP 2 model (~350 MB) on first run.
> It caches it locally so subsequent runs are instant.

### 3. Run the server

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## Usage

1. Select your region from the dropdown (default: Maharashtra, India)
2. Drop a bird photo or click to browse
3. Click **Identify Bird**
4. See: species name, confidence score, taxonomy, eBird profile,
   Wikipedia summary, conservation status, and recent local sightings

---

## Changing your region

Edit the `region-select` dropdown in `static/index.html` or just
select from the dropdown in the UI. Common codes:

| Code   | Region              |
|--------|---------------------|
| IN-MH  | Maharashtra, India  |
| IN     | All of India        |
| IN-DL  | Delhi               |
| IN-KA  | Karnataka           |
| US     | United States       |
| GB     | United Kingdom      |
| AU     | Australia           |

Full list: https://ebird.org/region

---

## APIs used

- **BioCLIP 2** — vision model for species classification (local, no key needed)
- **eBird API** — taxonomy, species codes, recent sightings (your key is in app.py)
- **iNaturalist API** — observation counts, conservation status (free, no key)
- **Wikipedia REST API** — bird descriptions (free, no key)