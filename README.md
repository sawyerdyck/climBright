# climBright

AI-powered rock climbing analysis app.

- **Hold Buddy** — Upload images of climbing holds to classify grip types (crimp, jug, sloper, etc.) with bounding box overlays showing detected holds.
- **Wall Analysis** — Upload wall photos to detect all holds, classify them, and get AI-generated climbing route suggestions with step-by-step coaching.
- Built with FastAPI, MongoDB, YOLOv8, ConvNeXt, Express, and Gemini AI.

<div style="display: flex; flex-wrap: wrap;">
  <img src="images/Screenshot 2026-01-31 194827.png" alt="Hold classification" style="width: 50%; padding: 5px;">
  <img src="images/Screenshot 2026-01-31 195233.png" alt="Wall route analysis" style="width: 50%; padding: 5px;">
</div>

## Features

- YOLOv8 hold detection with bounding boxes
- ConvNeXt-based grip type classification
- Reach-constrained greedy route pathfinding (local) or Gemini AI coaching
- Dark/light mode with system preference detection
- Responsive design (mobile-friendly)
- User authentication (JWT sessions)

## Quick Start (Local)

### One-command start

```powershell
# Windows PowerShell
.\start.ps1

# Stop everything
.\stop.ps1
```

```bash
# macOS / Linux
chmod +x start.sh stop.sh
./start.sh

# Stop everything
./stop.sh
```

This boots MongoDB, FastAPI (AI models), and Express (frontend) in one go. Open **http://127.0.0.1:3000/** when it reports all services running.

### Manual start (step by step)

### Prerequisites

- **MongoDB** (`mongod`) — [Install guide](https://www.mongodb.com/docs/manual/installation/)
- **Python 3.10+** with pip
- **Node.js 18+** with npm

### 1. Model weights

Place these files in the repo root:

| Model | Default path | Env var override |
|-------|-------------|-----------------|
| ConvNeXt classifier | `best_convnext_two_phase.pt` | `CONVNEXT_MODEL_PATH` |
| YOLO detector | `runs/detect/train2/weights/best.pt` | `YOLO_MODEL_PATH` |

### 2. Python environment

```bash
python -m venv env

# macOS/Linux
source env/bin/activate

# Windows PowerShell
.\env\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

### 3. Start MongoDB

```bash
mongod --dbpath ./db/mongo --bind_ip 127.0.0.1 --port 2701
```

### 4. Start FastAPI (AI model server)

```bash
uvicorn main:app --reload --port 9000
```

### 5. Start the web app

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
MONGODB_URI=mongodb://127.0.0.1:2701/climbright
FASTAPI_URL=http://127.0.0.1:9000/classifier/upload
JWT_SECRET=your-secret-here
PORT=3000
```

```bash
npm start
```

Open **http://127.0.0.1:3000/** — register an account and start uploading.

### 6. Smoke test

- Register / log in
- Go to `/holds` and upload a JPG/PNG — you should see bounding boxes and classification results
- Go to `/wall` and upload a wall photo — you should see hold markers, route overlay, and a coach summary

Optional API test:

```bash
B64=$(base64 -i sample.jpg | tr -d '\n')
curl -s http://127.0.0.1:9000/classifier/upload \
  -H 'Content-Type: application/json' \
  -d "{\"filename\":\"sample.jpg\",\"content_type\":\"image/jpeg\",\"data\":\"$B64\"}" \
  | python -m json.tool
```

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Browser    │────▶│   Express    │────▶│   MongoDB    │
│  (frontend)  │     │  (port 3000) │     │ (port 2701)  │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       │  image upload (base64)
       ▼
┌──────────────┐
│   FastAPI    │  YOLOv8 detect → ConvNeXt classify
│  (port 9000) │
└──────────────┘
       │
       ▼
┌──────────────┐
│  Pathfinder  │  Gemini AI coach (if API key set)
│  (Python)    │  or local greedy routing fallback
└──────────────┘
```

### Model Pipeline

1. **YOLOv8 detector** — finds hold bounding boxes in the image
2. **ConvNeXt classifier** — classifies each cropped hold into grip types
3. **Pathfinder** — generates climbing routes from detected holds:
   - **Gemini mode** (if `GEMINI_API_KEY` set): sends image + hold data to Gemini for coaching
   - **Local mode** (fallback): reach-constrained greedy routing with difficulty scoring

### Pathfinder Algorithm (Local)

- Greedy bottom-to-top traversal within realistic reach distance (0.30 normalized)
- Lateral movement penalized to prefer direct upward paths
- Route B uses harder hold types with bigger reach allowance
- Difficulty scored from hold-type weights + gap distances + lateral variance

---

## Training Models

### ConvNeXt Hold Classifier

Dataset: [Indoor Climbing Gym Hold Classification](https://www.kaggle.com/datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/data)

```bash
# Convert dataset to folder structure
python convert_to_folders.py

# Train (two-phase: frozen head → full fine-tune)
python two_phase_train.py
```

### YOLOv8 Hold Detector

```bash
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=gpu
```

### Combined Inference

```bash
python detect_and_classify.py -i "path/to/image.jpg" -y "path/to/yolo.pt" -c "path/to/convnext.pt"
```

Options: `-i` image, `-y` YOLO model, `-c` ConvNeXt model, `--conf` threshold, `--padding` box padding, `--no-save` skip visualization.

---

## Project Structure

```
climBright/
├── main.py                  # FastAPI app entry point
├── routers/classifier.py    # /classifier/upload endpoint
├── pathfinder.py            # Route generation (Gemini + local fallback)
├── detect_and_classify.py   # YOLOv8 + ConvNeXt pipeline
├── two_phase_train.py       # ConvNeXt training script
├── requirements.txt         # Python dependencies
├── frontend/
│   ├── server.js            # Express server (auth, static, API proxy)
│   ├── holds.html/js        # Hold Buddy page
│   ├── wall.html/js         # Wall Analysis page
│   ├── login.html           # Login page
│   ├── register.html        # Registration page
│   ├── styles.css           # Dark/light theme CSS
│   ├── theme.js             # Theme toggle logic
│   ├── auth.js              # Auth form handlers
│   ├── favicon.svg          # App favicon
│   ├── .env                 # Environment config (not committed)
│   └── src/
│       ├── routes/          # Express API routes (auth, images, wall)
│       ├── models/          # Mongoose schemas
│       └── middleware/      # JWT auth middleware
└── runs/detect/             # YOLO training outputs
```

---

## Environment Variables

### Frontend (`frontend/.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | Yes |
| `FASTAPI_URL` | FastAPI classifier endpoint URL | Yes |
| `JWT_SECRET` | Secret for JWT session tokens | Yes |
| `PORT` | Express server port (default: 3000) | No |
| `FRONTEND_ORIGIN` | Allowed CORS origin | No |

### Backend (root `.env` or shell)

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Enables Gemini AI coaching | No |
| `CONVNEXT_MODEL_PATH` | Path to ConvNeXt weights | No (defaults to `best_convnext_two_phase.pt`) |
| `YOLO_MODEL_PATH` | Path to YOLO weights | No (defaults to `runs/detect/train2/weights/best.pt`) |
| `PYTHON_BIN` | Python binary for pathfinder subprocess | No |

---

## License

See [LICENSE](LICENSE).
