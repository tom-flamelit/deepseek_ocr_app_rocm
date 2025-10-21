# 🚀 DeepSeek OCR - React + FastAPI

Modern OCR web application powered by DeepSeek-OCR with a stunning React frontend and FastAPI backend.

> **Note**: This was a quickly vibe-coded project to test out DeepSeek-OCR! The initial version targeted an RTX 5090, but this fork refreshes the stack for **AMD GPUs running ROCm 6.1**. The "Find" mode grounding boxes aren't quite working yet - probably my fault in not interpreting the dimensions correctly, but the core OCR functionality is pretty nice so far.

## Quick Start

```bash
docker compose up --build
```

Then open:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

### 4 OCR Modes
- **Plain OCR** - Raw text extraction
- **Describe** - Generate image descriptions
- **Find** - Locate specific terms (grounding boxes WIP)
- **Freeform** - Custom prompts for anything

### UI Features
- 🎨 Glass morphism design with animated gradients
- 🎯 Drag & drop file upload
- 📦 Grounding box visualization (WIP - dimensions need fixing)
- ✨ Smooth animations (Framer Motion)
- 📋 Copy/Download results
- 🎛️ Advanced settings dropdown
- 📝 Markdown rendering for formatted output

## Tech Stack

- **Frontend**: React 18 + Vite 5 + TailwindCSS 3 + Framer Motion 11
- **Backend**: FastAPI + PyTorch + Transformers 4.46 + DeepSeek-OCR
- **Server**: Nginx (reverse proxy)
- **Container**: Docker + Docker Compose with multi-stage builds
- **GPU**: AMD ROCm 6.1 support (tested on Radeon PRO W7800)

## Project Structure

```
deepseek-ocr/
├── backend/           # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/          # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── nginx.conf
│   └── Dockerfile
├── models/            # Model cache
└── docker-compose.yml
```

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Requirements

- Docker & Docker Compose
- AMD GPU with ROCm 6.1 support (tested on Radeon PRO W7800)
- ROCm drivers/runtime on the host (`/dev/kfd` & `/dev/dri` exposed to containers)
- ~8-12GB VRAM for model

## Known Issues

- 📦 **Find mode grounding boxes**: Not rendering correctly - likely dimension scaling issue in the canvas overlay logic. Boxes are detected and returned by the backend, but the frontend visualization needs work.

## ROCm Notes

- The backend container is built from `rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.5.1`, so no extra PyTorch install steps are required.
- Ensure the host user has access to `/dev/kfd` and `/dev/dri`. Adding your user to the `video` and `render` groups usually does the trick.
- If your GPU reports an older GFX IP, set `HSA_OVERRIDE_GFX_VERSION` in `.env` or the compose file (for example `11.0.0` for RDNA3).
- Override `TORCH_DEVICE` or `TORCH_DTYPE` environment variables if you need to force a specific device/dtype (defaults are auto-detected).

## API Usage

### POST /api/ocr

**Parameters:**
- `image` (file, required)
- `mode` (string): plain_ocr | describe | find_ref | freeform
- `prompt` (string): Custom prompt for freeform mode
- `grounding` (bool): Enable bounding boxes (auto-enabled for find_ref)
- `find_term` (string): Term to locate in find_ref mode
- `base_size` (int): Base processing size (default: 1024)
- `image_size` (int): Image size (default: 640)
- `crop_mode` (bool): Enable crop mode (default: true)

**Response:**
```json
{
  "success": true,
  "text": "Extracted text...",
  "boxes": [{"label": "field", "box": [x1, y1, x2, y2]}],
  "image_dims": {"w": 1920, "h": 1080},
  "metadata": {...}
}
```

## Troubleshooting

### GPU not detected
```bash
rocminfo
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.5.1 rocminfo
```

### Port conflicts
```bash
sudo lsof -i :3000
sudo lsof -i :8000
```

### Frontend build issues
```bash
cd frontend
rm -rf node_modules package-lock.json
docker-compose build frontend
```

## License

This project uses the DeepSeek-OCR model. Refer to the model's license terms.
