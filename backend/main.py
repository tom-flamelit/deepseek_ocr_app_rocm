import os
import re
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import uvicorn

# -----------------------------
# Lifespan context for model loading
# -----------------------------
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer
    
    # Environment setup
    os.environ.pop("TRANSFORMERS_CACHE", None)
    MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
    HF_HOME = os.environ.get("HF_HOME", "/models")
    os.makedirs(HF_HOME, exist_ok=True)
    
    # Load model
    print(f"🚀 Loading {MODEL_NAME}...")

    requested_device = os.environ.get("TORCH_DEVICE")
    if requested_device:
        device = requested_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype_env = os.environ.get("TORCH_DTYPE", "").strip().lower()
    dtype_lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_lookup.get(dtype_env)

    if torch_dtype is None:
        if device == "cpu":
            torch_dtype = torch.float32
        else:
            is_bf16_supported = False
            if torch.cuda.is_available():
                try:
                    is_bf16_supported = bool(torch.cuda.is_bf16_supported())
                except AttributeError:
                    is_bf16_supported = False
            torch_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    print(f"📦 Using device='{device}' with dtype='{torch_dtype}'")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    ).eval().to(device)
    
    # Pad token setup
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    
    print("✅ Model loaded and ready!")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR API",
    description="Blazing fast OCR with DeepSeek-OCR model 🔥",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(
    mode: str,
    user_prompt: str,
    grounding: bool,
    find_term: Optional[str],
    schema: Optional[str],
    include_caption: bool,
) -> str:
    """Build the prompt based on mode"""
    parts: List[str] = ["<image>"]
    mode_requires_grounding = mode in {"find_ref", "layout_map", "pii_redact"}
    if grounding or mode_requires_grounding:
        parts.append("<|grounding|>")

    instruction = ""
    if mode == "plain_ocr":
        instruction = "Free OCR. Only output the raw text."
    elif mode == "markdown":
        instruction = "Convert the document to markdown."
    elif mode == "tables_csv":
        instruction = (
            "Extract every table and output CSV only. "
            "Use commas, minimal quoting. If multiple tables, separate with a line containing '---'."
        )
    elif mode == "tables_md":
        instruction = "Extract every table as GitHub-flavored Markdown tables. Output only the tables."
    elif mode == "kv_json":
        schema_text = schema.strip() if schema else "{}"
        instruction = (
            "Extract key fields and return strict JSON only. "
            f"Use this schema (fill the values): {schema_text}"
        )
    elif mode == "figure_chart":
        instruction = (
            "Parse the figure. First extract any numeric series as a two-column table (x,y). "
            "Then summarize the chart in 2 sentences. Output the table, then a line '---', then the summary."
        )
    elif mode == "find_ref":
        key = (find_term or "").strip() or "Total"
        instruction = f"Locate <|ref|>{key}<|/ref|> in the image."
    elif mode == "layout_map":
        instruction = (
            'Return a JSON array of blocks with fields {"type":["title","paragraph","table","figure"],'
            '"box":[x1,y1,x2,y2]}. Do not include any text content.'
        )
    elif mode == "pii_redact":
        instruction = (
            'Find all occurrences of emails, phone numbers, postal addresses, and IBANs. '
            'Return a JSON array of objects {label, text, box:[x1,y1,x2,y2]}.'
        )
    elif mode == "multilingual":
        instruction = "Free OCR. Detect the language automatically and output in the same script."
    elif mode == "describe":
        instruction = "Describe this image concisely in 2-3 sentences. Focus on visible key elements."
    elif mode == "freeform":
        instruction = user_prompt.strip() if user_prompt else "OCR this image."
    else:
        instruction = "OCR this image."

    if include_caption and mode not in {"describe"}:
        instruction = instruction + "\nThen add a one-paragraph description of the image."

    parts.append(instruction)
    return "\n".join(parts)

# -----------------------------
# Grounding parser
# -----------------------------
DET_BLOCK = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*\[\s*\[\s*(?P<coords>[^\]]+?)\s*\]\s*\]\s*<\|/det\|>",
    re.DOTALL,
)

def clean_grounding_text(text: str) -> str:
    """Remove grounding tags from text for display, keeping labels"""
    # Replace <|ref|>label<|/ref|><|det|>[[...]]<|/det|> with just "label"
    cleaned = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[\s*\[[^\]]+\]\s*\]\s*<\|/det\|>",
        r"\1",
        text,
        flags=re.DOTALL
    )
    # Also remove any standalone grounding tags
    cleaned = re.sub(r"<\|grounding\|>", "", cleaned)
    return cleaned.strip()

def parse_detections(text: str) -> List[Dict[str, Any]]:
    """Parse grounding boxes from text"""
    boxes: List[Dict[str, Any]] = []
    for m in DET_BLOCK.finditer(text or ""):
        label = m.group("label").strip()
        coords = [c.strip() for c in m.group("coords").split(",")]
        try:
            nums = list(map(float, coords[:4]))
        except Exception:
            continue
        if len(nums) == 4:
            boxes.append({"label": label, "box": nums})
    return boxes

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR API is running! 🚀", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/ocr")
async def ocr_inference(
    image: UploadFile = File(...),
    mode: str = Form("plain_ocr"),
    prompt: str = Form(""),
    grounding: bool = Form(False),
    include_caption: bool = Form(False),
    find_term: Optional[str] = Form(None),
    schema: Optional[str] = Form(None),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
    test_compress: bool = Form(False),
):
    """
    Perform OCR inference on uploaded image
    
    - **image**: Image file to process
    - **mode**: OCR mode (plain_ocr, markdown, tables_csv, etc.)
    - **prompt**: Custom prompt for freeform mode
    - **grounding**: Enable grounding boxes
    - **include_caption**: Add image description
    - **find_term**: Term to find (for find_ref mode)
    - **schema**: JSON schema (for kv_json mode)
    - **base_size**: Base processing size
    - **image_size**: Image size parameter
    - **crop_mode**: Enable crop mode
    - **test_compress**: Test compression
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Build prompt
    prompt_text = build_prompt(
        mode=mode,
        user_prompt=prompt,
        grounding=grounding,
        find_term=find_term,
        schema=schema,
        include_caption=include_caption,
    )
    
    tmp_img = None
    out_dir = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_img = tmp.name
        
        # Get original dimensions
        try:
            with Image.open(tmp_img) as im:
                orig_w, orig_h = im.size
        except Exception:
            orig_w = orig_h = None
        
        out_dir = tempfile.mkdtemp(prefix="dsocr_")
        
        # Run inference
        res = model.infer(
            tokenizer,
            prompt=prompt_text,
            image_file=tmp_img,
            output_path=out_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=test_compress,
            eval_mode=True,
        )
        
        # Normalize response
        if isinstance(res, str):
            text = res.strip()
        elif isinstance(res, dict) and "text" in res:
            text = str(res["text"]).strip()
        elif isinstance(res, (list, tuple)):
            text = "\n".join(map(str, res)).strip()
        else:
            text = ""
        
        # Fallback: check output file
        if not text:
            mmd = os.path.join(out_dir, "result.mmd")
            if os.path.exists(mmd):
                with open(mmd, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
        if not text:
            text = "No text returned by model."
        
        # Parse grounding boxes
        boxes = parse_detections(text) if ("<|det|>" in text or "<|ref|>" in text) else []
        
        # Clean grounding tags from display text, but keep the labels
        display_text = clean_grounding_text(text) if ("<|ref|>" in text or "<|grounding|>" in text) else text
        
        # If display text is empty after cleaning but we have boxes, show the labels
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])
        
        return JSONResponse({
            "success": True,
            "text": display_text,
            "boxes": boxes,
            "image_dims": {"w": orig_w, "h": orig_h},
            "metadata": {
                "mode": mode,
                "grounding": grounding or (mode in {"find_ref","layout_map","pii_redact"}),
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    
    finally:
        if tmp_img:
            try:
                os.remove(tmp_img)
            except Exception:
                pass
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
