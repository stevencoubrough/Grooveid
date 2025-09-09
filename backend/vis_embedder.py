import io, requests, numpy as np, torch, open_clip
from PIL import Image
from .vec_cache import load as cache_load, save as cache_save

# Load once per process
DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
MODEL_PRETRAIN = "openai"   # widely available
_model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAIN, device=DEVICE)
_model.eval()

def _to_image(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@torch.no_grad()
def embed_image_bytes(img_bytes: bytes) -> np.ndarray:
    im = _to_image(img_bytes)
    ten = _preprocess(im).unsqueeze(0).to(DEVICE)
    feat = _model.encode_image(ten)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def embed_image_url(url: str) -> np.ndarray | None:
    vec = cache_load(url)
    if vec is not None:
        return vec
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        vec = embed_image_bytes(r.content)
        cache_save(url, vec)
        return vec
    except Exception:
        return None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)
