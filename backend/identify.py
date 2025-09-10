# backend/identify.py
# GrooveID — Complete version with perfect matching pipeline

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, io, time, base64, requests, logging
import difflib
from collections import Counter
import hashlib
import colorsys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 3rd-party ----------
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

# Optional imports with fallbacks
try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
    logger.info("CLIP libraries loaded successfully")
except ImportError as e:
    CLIP_AVAILABLE = False
    logger.warning(f"CLIP not available: {e}")
    torch = None
    open_clip = None

# Perfect matching optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - visual analysis disabled")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available - multi-OCR disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - semantic matching disabled")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

# Supabase with error handling
supabase = None
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    else:
        logger.warning("Supabase credentials not found")
except Exception as e:
    logger.warning(f"Supabase not available: {e}")
    supabase = None

# ---------- Config ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

DGS_API = "https://api.discogs.com"
DGS_UA = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

# Regex
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

router = APIRouter()

# ---------- Models ----------
class IdentifyCandidate(BaseModel):
    source: str
    release_id: Optional[int] = None
    master_id: Optional[int] = None
    discogs_url: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    label: Optional[str] = None
    year: Optional[str] = None
    cover_url: Optional[str] = None
    score: float
    note: Optional[str] = None

class IdentifyResponse(BaseModel):
    candidates: List[IdentifyCandidate]

# ---------- PERFECT MATCHING: ADVANCED COMPUTER VISION ----------
def extract_color_palette(image_bytes: bytes) -> List[str]:
    """Extract dominant colors from record label for visual matching"""
    if not CV2_AVAILABLE:
        return []
    
    try:
        import cv2
        import numpy as np
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize for processing speed
        img = cv2.resize(img, (150, 150))
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape for K-means clustering
        pixels = img_rgb.reshape(-1, 3)
        
        # K-means to find dominant colors
        if SKLEARN_AVAILABLE:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            
            colors = []
            for color in kmeans.cluster_centers_:
                # Convert to hex
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
                colors.append(hex_color)
            
            return colors
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
    
    return []

def detect_label_shape(image_bytes: bytes) -> Dict[str, any]:
    """Detect physical characteristics of the label"""
    if not CV2_AVAILABLE:
        return {'estimated_size': '12inch'}
    
    try:
        import cv2
        import numpy as np
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect circles (for center hole and label edge)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=300)
        
        characteristics = {
            'has_center_hole': False,
            'label_shape': 'unknown',
            'estimated_size': '12inch'  # default
        }
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) >= 2:
                # Sort by radius
                circles = sorted(circles, key=lambda x: x[2])
                characteristics['has_center_hole'] = True
                
                # Estimate size based on hole-to-label ratio
                if len(circles) >= 2:
                    hole_radius = circles[0][2]
                    label_radius = circles[-1][2]
                    ratio = hole_radius / label_radius if label_radius > 0 else 0
                    
                    if ratio > 0.15:
                        characteristics['estimated_size'] = '7inch'
                    elif ratio > 0.08:
                        characteristics['estimated_size'] = '12inch'
                    else:
                        characteristics['estimated_size'] = '10inch'
        
        return characteristics
    except Exception as e:
        logger.error(f"Shape detection failed: {e}")
        return {'estimated_size': '12inch'}

# ---------- FUZZY STRING MATCHING ----------
def fuzzy_match_score(text1: str, text2: str, threshold: float = 0.6) -> float:
    """Calculate similarity between two strings using difflib"""
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def find_best_fuzzy_match(target: str, candidates: List[str], threshold: float = 0.6) -> Tuple[str, float]:
    """Find the best fuzzy match from a list of candidates"""
    best_match = ""
    best_score = 0.0
    
    for candidate in candidates:
        score = fuzzy_match_score(target, candidate, threshold)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

# ---------- OCR ERROR CORRECTION ----------
def correct_ocr_errors(text: str) -> str:
    """Correct common OCR errors using known patterns"""
    corrections = {
        # Common character misreads
        'TKON': 'TRON', 'IRON': 'TRON', 'KRON': 'TRON',
        'WAZ': 'WAX', 'WOX': 'WAX',
        'AC1D': 'ACID', 'ACOD': 'ACID',
        'WAR9': 'WARP',
        '0O1': '001', 'OO1': '001',
        'V0L': 'VOL', 'VOLURNE': 'VOLUME',
        'N1NJA': 'NINJA', 'HOSP1TAL': 'HOSPITAL',
    }
    
    corrected = text.upper()
    for error, correction in corrections.items():
        corrected = corrected.replace(error, correction)
    
    return corrected

def enhance_ocr_text(lines: List[str]) -> List[str]:
    """Apply OCR error correction to all lines"""
    enhanced = []
    for line in lines:
        corrected = correct_ocr_errors(line)
        enhanced.append(corrected)
        # Also keep original if different
        if corrected != line.upper():
            enhanced.append(line)
    
    return list(dict.fromkeys(enhanced))  # Remove duplicates

# ---------- MULTI-OCR ENGINE APPROACH ----------
def multi_ocr_extraction(image_bytes: bytes) -> Dict[str, List[str]]:
    """Use multiple OCR engines for better text extraction"""
    results = {'google': [], 'tesseract': [], 'easyocr': []}
    
    # Google Vision (already implemented)
    try:
        v = call_vision_full(image_bytes)
        if v.get("textAnnotations"):
            google_text = v["textAnnotations"][0].get("description", "")
            results['google'] = [ln.strip() for ln in google_text.splitlines() if ln.strip()]
    except:
        pass
    
    # Tesseract OCR
    if TESSERACT_AVAILABLE:
        try:
            import pytesseract
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_bytes))
            # Multiple Tesseract configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 13', # Raw line
            ]
            
            tesseract_lines = []
            for config in configs:
                text = pytesseract.image_to_string(img, config=config)
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                tesseract_lines.extend(lines)
            
            results['tesseract'] = list(dict.fromkeys(tesseract_lines))
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
    
    # EasyOCR (if available)
    if EASYOCR_AVAILABLE:
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            easyocr_results = reader.readtext(image_bytes)
            results['easyocr'] = [result[1] for result in easyocr_results if result[2] > 0.5]
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
    
    return results

def consensus_ocr(multi_ocr_results: Dict[str, List[str]]) -> List[str]:
    """Create consensus from multiple OCR engines"""
    all_lines = []
    for engine, lines in multi_ocr_results.items():
        all_lines.extend(lines)
    
    # Use fuzzy matching to group similar lines
    consensus_lines = []
    used_indices = set()
    
    for i, line1 in enumerate(all_lines):
        if i in used_indices:
            continue
            
        similar_lines = [line1]
        used_indices.add(i)
        
        for j, line2 in enumerate(all_lines[i+1:], i+1):
            if j in used_indices:
                continue
                
            if fuzzy_match_score(line1, line2) > 0.8:
                similar_lines.append(line2)
                used_indices.add(j)
        
        # Take the most common or longest version
        best_line = max(similar_lines, key=len)
        consensus_lines.append(best_line)
    
    return consensus_lines

# ---------- GENRE-SPECIFIC PATTERN DETECTION ----------
def detect_music_genre(lines: List[str]) -> List[str]:
    """Detect music genre from OCR text to apply genre-specific search patterns"""
    all_text = ' '.join(lines).lower()
    
    genre_indicators = {
        'techno': ['techno', 'tech', 'minimal', 'detroit', 'berliner', 'industrial'],
        'house': ['house', 'deep', 'vocal', 'jackin', 'chicago', 'garage'],
        'trance': ['trance', 'progressive', 'uplifting', 'psy', 'goa'],
        'drum_and_bass': ['drum', 'bass', 'dnb', 'd&b', 'jungle', 'breakbeat'],
        'hardcore': ['hardcore', 'gabber', 'speedcore', 'terror', 'frenchcore'],
        'acid': ['acid', '303', 'tb303', 'squelch'],
        'ambient': ['ambient', 'chillout', 'downtempo', 'dub', 'space'],
        'breakbeat': ['breakbeat', 'breaks', 'big beat', 'chemical'],
        'electro': ['electro', 'miami', 'bass', 'freestyle']
    }
    
    detected_genres = []
    for genre, indicators in genre_indicators.items():
        if any(indicator in all_text for indicator in indicators):
            detected_genres.append(genre)
    
    return detected_genres

def get_genre_specific_patterns(genres: List[str], label: str, catno: str) -> List[Dict[str, str]]:
    """Generate genre-specific search patterns"""
    patterns = []
    
    for genre in genres:
        if genre == 'techno':
            patterns.extend([
                {"q": f"{label} minimal"} if label else {"q": "minimal"},
                {"q": f"{label} detroit"} if label else {"q": "detroit"},
                {"genre": "Electronic", "style": "Techno"},
            ])
        elif genre == 'house':
            patterns.extend([
                {"q": f"{label} deep house"} if label else {"q": "deep house"},
                {"q": f"{label} vocal"} if label else {"q": "vocal"},
                {"genre": "Electronic", "style": "House"},
            ])
        elif genre == 'drum_and_bass':
            patterns.extend([
                {"q": f"{label} jungle"} if label else {"q": "jungle"},
                {"q": f"{label} liquid"} if label else {"q": "liquid"},
                {"genre": "Electronic", "style": "Drum n Bass"},
            ])
        elif genre == 'acid':
            patterns.extend([
                {"q": f"{label} 303"} if label else {"q": "303"},
                {"q": f"{label} acid house"} if label else {"q": "acid house"},
                {"genre": "Electronic", "style": "Acid"},
            ])
    
    return patterns

# ---------- ADVANCED METADATA EXTRACTION ----------
def extract_advanced_metadata(lines: List[str]) -> Dict[str, any]:
    """Enhanced metadata extraction with fuzzy matching and error correction"""
    # Apply OCR error correction first
    enhanced_lines = enhance_ocr_text(lines)
    all_text = ' '.join(enhanced_lines).lower()
    
    metadata = {
        'label': None,
        'catno': None, 
        'artist': None,
        'tracks': [],
        'genres': [],
        'confidence_scores': {}
    }
    
    # Detect genres first
    metadata['genres'] = detect_music_genre(enhanced_lines)
    
    # Enhanced label detection with known electronic music labels
    known_labels = [
        'TRON', 'WAX', 'ACID', 'WARP', 'NINJA', 'HOSPITAL', 'METALHEADZ', 'GOOD LOOKING',
        'REINFORCED', 'MOVING SHADOW', 'SUBURBAN BASE', 'FORMATION', 'CERTIFICATE 18',
        'HARDLEADERS', 'MOKUM', 'THUNDERDOME', 'ID&T', 'BONZAI', 'R&S', 'APOLLO',
        'SOMA', 'PEACEFROG', 'HOOJ', 'GLOBAL UNDERGROUND', 'BEDROCK', 'SYSTEMATIC',
        'COCOON', 'MINUS', 'M_NUS', 'KOMPAKT', 'PERLON', 'OSTGUT TON'
    ]
    
    # Try to match against known labels with fuzzy matching
    words_in_text = [word for word in all_text.split() if len(word) >= 3]
    
    for known_label in known_labels:
        for word in words_in_text:
            similarity = fuzzy_match_score(word, known_label.lower())
            if similarity > 0.7:  # 70% similarity threshold
                metadata['label'] = known_label
                metadata['confidence_scores']['label'] = similarity
                break
        if metadata['label']:
            break
    
    # If no fuzzy match, fall back to pattern matching
    if not metadata['label']:
        patterns = [
            r'([a-z]{3,8})\s+(\d{2,3})(?:\s|$)',
            r'([a-z]{3,8})-(\d{2,3})(?:\s|$)',
            r'([a-z]{3,8})(\d{2,3})(?:\s|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, all_text)
            for m in matches:
                potential_label = m.group(1).upper()
                potential_catno = m.group(2).zfill(3)
                
                excluded_words = ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'THAT', 'WITH', 'FROM', 'LTD', 'INC']
                phone_context = any(phone in all_text[max(0, m.start()-20):m.end()+20] for phone in ['0181', '020', '01', 'phone', 'tel'])
                
                if (potential_label not in excluded_words and 
                    len(potential_label) >= 3 and 
                    not phone_context):
                    metadata['label'] = potential_label
                    metadata['catno'] = potential_catno
                    metadata['confidence_scores']['label'] = 0.8
                    break
            if metadata['label']:
                break
    
    # Enhanced artist detection
    for i, line in enumerate(enhanced_lines[:int(len(enhanced_lines) * 0.4)]):
        line_clean = line.strip()
        words = line_clean.split()
        
        # Skip obvious non-artist lines
        skip_indicators = ['vol', 'volume', 'side', 'records', 'music', 'label', 'catalog', 
                          'manufactured', 'distributed', 'copyright', 'published', 'written', 'produced']
        if any(indicator in line.lower() for indicator in skip_indicators):
            continue
        
        # Artist name patterns
        if (2 <= len(words) <= 4 and 
            5 <= len(line_clean) <= 40 and
            not re.search(r'\d{3,}', line_clean)):
            
            if (any(w[0].isupper() for w in words if w.isalpha()) or 
                all(w.isupper() for w in words if w.isalpha())):
                metadata['artist'] = line_clean
                metadata['confidence_scores']['artist'] = 0.7
                break
    
    # Track extraction
    for line in enhanced_lines:
        line_lower = line.lower().strip()
        
        track_patterns = [
            r'^[ab]\d*\s*[:\-\.]\s*(.+)',
            r'^\d+\s*[:\-\.]\s*(.+)',
            r'^side\s+[ab]\s*[:\-]\s*(.+)',
        ]
        
        for pattern in track_patterns:
            m = re.match(pattern, line_lower)
            if m:
                track_name = m.group(1).strip()
                # Clean track name
                track_name = re.sub(r'\s*-?\s*\d+rpm\s*$', '', track_name)
                if len(track_name) > 2:
                    metadata['tracks'].append(track_name.title())
                break
    
    return metadata

# ---------- CATALOG NUMBER PATTERN ANALYSIS ----------
def analyze_catalog_patterns(label: str, catno: str) -> List[Dict[str, str]]:
    """Generate searches based on label-specific catalog patterns"""
    patterns = []
    
    if not label or not catno:
        return patterns
    
    # Label-specific catalog patterns
    label_patterns = {
        'WARP': ['WARP{:03d}', 'WARPCD{:03d}', 'WARPLP{:03d}'],
        'NINJA': ['NINJA{:03d}', 'ZEN{:02d}', 'ZENCD{:02d}'],
        'HOSPITAL': ['NHS{:03d}', 'HOSP{:03d}'],
        'METALHEADZ': ['METH{:03d}', 'METHCD{:03d}'],
        'MOVING SHADOW': ['SHADOW{:02d}', 'ASHADOW{:02d}'],
        'R&S': ['RS{:05d}', 'R&S{:02d}'],
        'PLUS 8': ['PLUS8{:03d}', '+8{:03d}'],
        'PEACEFROG': ['PFG{:03d}', 'PFGCD{:03d}'],
    }
    
    catno_num = int(re.search(r'\d+', catno).group()) if re.search(r'\d+', catno) else 0
    
    if label.upper() in label_patterns:
        for pattern in label_patterns[label.upper()]:
            try:
                formatted_catno = pattern.format(catno_num)
                patterns.append({"catno": formatted_catno})
                patterns.append({"q": formatted_catno})
            except:
                continue
    
    # Generic patterns
    patterns.extend([
        {"catno": f"{label}{catno:0>3}"},
        {"catno": f"{label} {catno:0>3}"},
        {"catno": f"{label}-{catno:0>3}"},
        {"q": f"{label}{catno:0>3}"},
    ])
    
    return patterns

# ---------- TRACK LISTING VERIFICATION ----------
def verify_track_listing(detected_tracks: List[str], candidate: IdentifyCandidate) -> float:
    """Verify candidate by fetching and comparing track listings"""
    try:
        if not detected_tracks or not candidate.release_id:
            return 0.5  # Neutral score
        
        # Fetch full release info
        release_info = fetch_discogs_release_json(candidate.release_id)
        if not release_info:
            return 0.5
        
        # Extract track names from Discogs
        discogs_tracks = []
        for track in release_info.get('tracklist', []):
            track_title = track.get('title', '')
            if track_title:
                discogs_tracks.append(track_title.lower().strip())
        
        if not discogs_tracks:
            return 0.5
        
        # Calculate track matching score
        matches = 0
        for detected_track in detected_tracks:
            detected_clean = detected_track.lower().strip()
            for discogs_track in discogs_tracks:
                # Use fuzzy matching for track names
                if fuzzy_match_score(detected_clean, discogs_track) > 0.7:
                    matches += 1
                    break
        
        # Calculate match ratio
        match_ratio = matches / max(len(detected_tracks), len(discogs_tracks))
        return match_ratio
    
    except Exception as e:
        logger.error(f"Track verification failed: {e}")
        return 0.5

# ---------- YEAR/ERA FILTERING ----------
def estimate_release_era(lines: List[str], image_bytes: bytes) -> Optional[int]:
    """Estimate release year from visual and text cues"""
    estimated_year = None
    
    # Look for explicit year mentions
    all_text = ' '.join(lines)
    year_patterns = [
        r'\b(19[89]\d|20[012]\d)\b',  # 1980-2029
        r'\(([12]\d{3})\)',           # (1995)
        r'©\s*([12]\d{3})',           # © 1995
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, all_text)
        if matches:
            try:
                year = int(matches[0])
                if 1980 <= year <= 2030:
                    estimated_year = year
                    break
            except:
                continue
    
    # Visual era estimation (if no explicit year found)
    if not estimated_year:
        try:
            # Analyze color palette for era clues
            colors = extract_color_palette(image_bytes)
            
            # Heuristic: Bright neon colors = 80s/90s, Minimal = 2000s+
            bright_colors = 0
            for color in colors:
                if color.startswith('#'):
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                    if s > 0.7 and v > 0.7:  # High saturation and brightness
                        bright_colors += 1
            
            if bright_colors >= 2:
                estimated_year = 1995  # Likely 90s
            else:
                estimated_year = 2005  # Likely 2000s+
        except:
            pass
    
    return estimated_year

# ---------- SEMANTIC MATCHING ----------
def semantic_similarity_search(detected_metadata: Dict, candidate_results: List[IdentifyCandidate]) -> List[IdentifyCandidate]:
    """Use semantic similarity for better matching"""
    if not SKLEARN_AVAILABLE:
        return candidate_results
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create text representations
        query_text = f"{detected_metadata.get('label', '')} {detected_metadata.get('artist', '')} {' '.join(detected_metadata.get('tracks', []))}"
        
        candidate_texts = []
        for candidate in candidate_results:
            candidate_text = f"{candidate.label or ''} {candidate.artist or ''} {candidate.title or ''}"
            candidate_texts.append(candidate_text)
        
        if not candidate_texts:
            return candidate_results
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        all_texts = [query_text] + candidate_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Apply semantic boost to scores
        for i, candidate in enumerate(candidate_results):
            semantic_boost = similarities[i] * 0.3  # Max 30% boost
            candidate.score = min(0.99, candidate.score + semantic_boost)
        
        return sorted(candidate_results, key=lambda x: x.score, reverse=True)
    
    except Exception as e:
        logger.error(f"Semantic similarity failed: {e}")
        return candidate_results

# ---------- INTELLIGENT SEARCH STRATEGY ----------
def generate_intelligent_searches(metadata: Dict[str, any], lines: List[str]) -> List[Dict[str, str]]:
    """Generate search attempts using advanced metadata and genre detection"""
    attempts = []
    
    label = metadata.get('label')
    catno = metadata.get('catno') 
    artist = metadata.get('artist')
    genres = metadata.get('genres', [])
    confidence_scores = metadata.get('confidence_scores', {})
    
    logger.info(f"Generating intelligent searches for: {metadata}")
    
    # Priority 1: High-confidence exact matches
    if label and catno and confidence_scores.get('label', 0) > 0.8:
        attempts.extend([
            {"label": label, "catno": catno},
            {"catno": f"{label}{catno}"},
            {"q": f"{label} {catno}"},
        ])
    
    # Priority 2: Artist-centric searches
    if artist and confidence_scores.get('artist', 0) > 0.6:
        attempts.extend([
            {"artist": artist},
            {"q": artist},
            {"artist": artist, "label": label} if label else {"artist": artist},
            {"artist": artist, "genre": "Electronic"},
        ])
    
    # Priority 3: Genre-specific patterns
    if genres:
        genre_patterns = get_genre_specific_patterns(genres, label, catno)
        attempts.extend(genre_patterns)
    
    # Priority 4: Fuzzy label searches (for OCR errors)
    if label:
        # Generate variations for common OCR errors
        label_variations = [label]
        if label == 'TRON':
            label_variations.extend(['IRON', 'KRON', 'TKON'])
        elif label == 'WAX':
            label_variations.extend(['WOX', 'WAZ'])
        elif label == 'ACID':
            label_variations.extend(['AC1D', 'ACOD'])
        
        for variation in label_variations:
            attempts.extend([
                {"q": variation},
                {"q": f"{variation} {catno}"} if catno else {"q": variation},
            ])
    
    # Priority 5: Underground/white label patterns
    underground_indicators = ['promo', 'white label', 'test pressing', 'advance', 'promotional']
    all_text = ' '.join(lines).lower()
    
    if any(indicator in all_text for indicator in underground_indicators):
        attempts.extend([
            {"q": "white label"},
            {"q": "promotional use only"},
            {"q": f"{label} promo"} if label else {"q": "promo"},
            {"artist": "Unknown Artist", "genre": "Electronic"},
        ])
    
    # Priority 6: Text-based fallback with fuzzy matching
    meaningful_lines = [line for line in lines if len(line) > 4 and 
                       not any(skip in line.lower() for skip in ['manufactured', 'distributed', 'copyright'])]
    
    if meaningful_lines:
        attempts.extend([
            {"q": " ".join(meaningful_lines[:3])[:120]},
            {"q": " ".join(meaningful_lines[:2])[:100]},
        ])
    
    # Remove duplicates
    seen = set()
    unique_attempts = []
    for search in attempts:
        search_key = str(sorted(search.items()))
        if search_key not in seen:
            unique_attempts.append(search)
            seen.add(search_key)
    
    return unique_attempts[:25]  # Increased limit for better coverage

# ---------- ADVANCED RESULT SCORING ----------
def calculate_advanced_score(result: IdentifyCandidate, metadata: Dict[str, any], search_index: int) -> float:
    """Calculate advanced relevance score for search results"""
    base_score = result.score or 0.70
    boost = 0.0
    
    label = metadata.get('label')
    artist = metadata.get('artist')
    genres = metadata.get('genres', [])
    confidence_scores = metadata.get('confidence_scores', {})
    
    # Search position boost (earlier = better)
    if search_index < 3:
        boost += 0.25
    elif search_index < 6:
        boost += 0.20
    elif search_index < 10:
        boost += 0.15
    elif search_index < 15:
        boost += 0.10
    
    # Exact label match boost
    if label and result.label:
        label_similarity = fuzzy_match_score(label, result.label)
        if label_similarity > 0.9:
            boost += 0.20 * confidence_scores.get('label', 1.0)
        elif label_similarity > 0.7:
            boost += 0.15 * confidence_scores.get('label', 1.0)
    
    # Exact artist match boost
    if artist and result.artist:
        artist_similarity = fuzzy_match_score(artist, result.artist)
        if artist_similarity > 0.9:
            boost += 0.25 * confidence_scores.get('artist', 1.0)
        elif artist_similarity > 0.7:
            boost += 0.20 * confidence_scores.get('artist', 1.0)
    
    # Genre relevance boost
    if result.title:
        title_lower = result.title.lower()
        for genre in genres:
            genre_indicators = {
                'techno': ['techno', 'minimal', 'detroit'],
                'house': ['house', 'deep', 'vocal'],
                'acid': ['acid', '303'],
                'drum_and_bass': ['jungle', 'drum', 'bass', 'dnb']
            }
            
            if genre in genre_indicators:
                if any(indicator in title_lower for indicator in genre_indicators[genre]):
                    boost += 0.15
                    break
    
    # Underground music indicators
    if result.title and any(word in result.title.lower() for word in ['vol', 'volume', 'ep', 'remix', 'edit']):
        boost += 0.10
    
    # Unknown artist boost for underground releases
    if result.artist and 'unknown' in result.artist.lower():
        boost += 0.10
    
    return min(0.99, base_score + boost)

# ---------- PERFECT MATCHING PIPELINE ----------
def perfect_matching_pipeline(image_bytes: bytes, basic_candidates: List[IdentifyCandidate]) -> List[IdentifyCandidate]:
    """Advanced pipeline for near-perfect matching"""
    
    # Stage 1: Multi-OCR consensus
    multi_ocr_results = multi_ocr_extraction(image_bytes)
    consensus_lines = consensus_ocr(multi_ocr_results)
    
    # Stage 2: Enhanced metadata with confidence
    enhanced_metadata = extract_advanced_metadata(consensus_lines)
    
    # Stage 3: Visual analysis
    visual_characteristics = detect_label_shape(image_bytes)
    estimated_year = estimate_release_era(consensus_lines, image_bytes)
    
    # Stage 4: Label-specific catalog patterns
    catalog_patterns = analyze_catalog_patterns(
        enhanced_metadata.get('label'), 
        enhanced_metadata.get('catno')
    )
    
    # Stage 5: Execute additional searches with catalog patterns
    additional_candidates = []
    for pattern in catalog_patterns:
        try:
            results = discogs_search(pattern)
            additional_candidates.extend(results)
        except:
            continue
    
    # Combine all candidates
    all_candidates = basic_candidates + additional_candidates
    
    # Stage 6: Advanced filtering and scoring
    enhanced_candidates = []
    for candidate in all_candidates:
        # Track listing verification
        track_score = verify_track_listing(
            enhanced_metadata.get('tracks', []), 
            candidate
        )
        
        # Year filtering (if we estimated a year)
        year_penalty = 0.0
        if estimated_year and candidate.year:
            try:
                candidate_year = int(candidate.year)
                year_diff = abs(estimated_year - candidate_year)
                if year_diff > 10:
                    year_penalty = 0.2  # Penalize if more than 10 years off
            except:
                pass
        
        # Size filtering
        size_bonus = 0.0
        if visual_characteristics.get('estimated_size') == '12inch':
            # Prefer 12" releases for detected 12" records
            if candidate.title and any(indicator in candidate.title.lower() 
                                     for indicator in ['12"', 'twelve', 'maxi']):
                size_bonus = 0.1
        
        # Apply all adjustments
        final_score = candidate.score + (track_score * 0.3) + size_bonus - year_penalty
        candidate.score = min(0.99, max(0.1, final_score))
        
        enhanced_candidates.append(candidate)
    
    # Stage 7: Semantic similarity re-ranking
    final_candidates = semantic_similarity_search(enhanced_metadata, enhanced_candidates)
    
    # Stage 8: Remove duplicates and sort
    seen_ids = set()
    unique_candidates = []
    for candidate in final_candidates:
        if candidate.release_id not in seen_ids:
            unique_candidates.append(candidate)
            seen_ids.add(candidate.release_id)
    
    return sorted(unique_candidates, key=lambda x: x.score, reverse=True)[:5]

# ---------- BASIC OCR FALLBACK ----------
def improved_ocr_fallback_with_ml(text, clean_lines, image_bytes):
    """Advanced OCR processing with machine learning-like features"""
    candidates = []
    
    logger.info("Starting advanced OCR identification")
    
    # Get all OCR text
    lines = ocr_lines(text)
    all_lines = list(dict.fromkeys([*lines, *clean_lines]))
    
    # Use advanced metadata extraction
    metadata = extract_advanced_metadata(all_lines)
    
    # Generate intelligent search attempts
    attempts = generate_intelligent_searches(metadata, all_lines)
    
    # Prioritize searches based on confidence scores
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    
    for attempt in attempts:
        # Check if this search uses high-confidence metadata
        uses_high_conf_label = (metadata.get('label') in str(attempt) and 
                               metadata.get('confidence_scores', {}).get('label', 0) > 0.8)
        uses_high_conf_artist = (metadata.get('artist') in str(attempt) and 
                                metadata.get('confidence_scores', {}).get('artist', 0) > 0.7)
        
        if uses_high_conf_label or uses_high_conf_artist:
            high_confidence.append(attempt)
        elif any(genre in str(attempt).lower() for genre in metadata.get('genres', [])):
            medium_confidence.append(attempt)
        else:
            low_confidence.append(attempt)
    
    # Execute in priority order
    ordered_attempts = high_confidence + medium_confidence + low_confidence
    
    for i, params in enumerate(ordered_attempts):
        try:
            logger.info(f"Advanced search {i+1}/{len(ordered_attempts)}: {params}")
            results = discogs_search(params)
            
            if results:
                # Apply advanced scoring
                for result in results:
                    result.score = calculate_advanced_score(result, metadata, i)
                
                candidates.extend(results)
                
                # Smart stopping criteria
                high_scoring = [c for c in results if c.score > 0.90]
                if high_scoring and len(candidates) >= 5:
                    logger.info(f"Found {len(high_scoring)} high-scoring matches, stopping")
                    break
                
                if len(candidates) >= 20:
                    break
                    
        except Exception as e:
            logger.error(f"Advanced search attempt {i+1} failed: {e}")
            continue
    
    # Sort by score and return top results
    candidates.sort(key=lambda x: x.score, reverse=True)
    logger.info(f"Advanced OCR returning {len(candidates)} candidates")
    return candidates[:5]

# ---------- Vision helpers ----------
def _vision_request(image_b64: str, features: List[Dict], ctx: Dict = None) -> dict:
    if not VISION_KEY:
        raise HTTPException(500, "GOOGLE_VISION_API_KEY not set")
    payload = {"requests": [{"image": {"content": image_b64}, "features": features}]}
    if ctx:
        payload["requests"][0]["imageContext"] = ctx
    
    try:
        r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
        if r.status_code != 200:
            logger.error(f"Vision API error {r.status_code}: {r.text[:200]}")
            raise HTTPException(502, f"Vision error {r.status_code}")
        return r.json().get("responses", [{}])[0]
    except requests.RequestException as e:
        logger.error(f"Vision API request failed: {e}")
        raise HTTPException(502, f"Vision API request failed: {str(e)}")

def call_vision_full(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        feats = [
            {"type": "WEB_DETECTION", "maxResults": 15},
            {"type": "TEXT_DETECTION", "maxResults": 10},
        ]
        ctx = {"webDetectionParams": {"includeGeoResults": True}}
        return _vision_request(b64, feats, ctx)
    except Exception as e:
        logger.error(f"Vision full call failed: {e}")
        return {}

def call_vision_doc(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return _vision_request(b64, [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}], {"languageHints": ["en"]})
    except Exception as e:
        logger.error(f"Vision doc call failed: {e}")
        return {}

def parse_discogs_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
            for it in web.get(key, []):
                if it.get("url"):
                    urls.append(it["url"])
        
        rel = mast = None
        discogs_url = None
        
        for u in urls:
            m = RE_RELEASE.search(u)
            if m:
                rel = int(m.group(1))
                discogs_url = u
                break
        
        if rel is None:
            for u in urls:
                m = RE_MASTER.search(u)
                if m:
                    mast = int(m.group(1))
                    discogs_url = u
                    break
        
        return rel, mast, discogs_url
    except Exception as e:
        logger.error(f"Error parsing discogs web results: {e}")
        return None, None, None

def ocr_lines(text_ann: List[dict]) -> List[str]:
    try:
        if not text_ann:
            return []
        raw = text_ann[0].get("description", "")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    except Exception as e:
        logger.error(f"Error extracting OCR lines: {e}")
        return []

# ---------- Enhanced image processing ----------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        s = int(min(w, h) * 0.85)
        cx, cy = w // 2, h // 2
        crop = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
        
        gray = ImageOps.grayscale(crop)
        sharp = gray.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=2))
        contrast = ImageEnhance.Contrast(sharp).enhance(2.8)
        result = ImageOps.equalize(contrast)
        
        return result
    except Exception as e:
        logger.error(f"Error in handwriting enhancement: {e}")
        return img

def handwriting_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        hw_img1 = enhance_for_handwriting(base_img)
        buf1 = io.BytesIO()
        hw_img1.save(buf1, format="PNG")
        v1 = call_vision_doc(buf1.getvalue())
        
        w, h = base_img.size
        upper_crop = base_img.crop((0, 0, w, h//2))
        hw_img2 = enhance_for_handwriting(upper_crop)
        buf2 = io.BytesIO()
        hw_img2.save(buf2, format="PNG")
        v2 = call_vision_doc(buf2.getvalue())
        
        extra_lines = []
        for v in [v1, v2]:
            if v.get("textAnnotations"):
                raw = v["textAnnotations"][0].get("description", "")
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                extra_lines.extend(lines)
        
        primary = [ln.strip() for ln in (text[0].get("description", "").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra_lines]))
        
        return [{"description": "\n".join(merged)}] if merged else text
    except Exception as e:
        logger.error(f"Error in handwriting merge: {e}")
        return text

def block_crop_reocr(image_bytes: bytes) -> List[str]:
    lines = []
    try:
        vdoc = call_vision_doc(image_bytes)
        base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        for page in vdoc.get("fullTextAnnotation", {}).get("pages", []):
            for block in page.get("blocks", []):
                verts = block.get("boundingBox", {}).get("vertices", [])
                if len(verts) == 4:
                    x = min(v.get("x", 0) for v in verts)
                    y = min(v.get("y", 0) for v in verts)
                    X = max(v.get("x", 0) for v in verts)
                    Y = max(v.get("y", 0) for v in verts)
                    
                    if X-x > 8 and Y-y > 8:
                        crop = base.crop((x, y, X, Y)).resize((int(2.0*(X-x)), int(2.0*(Y-y))))
                        enhanced_crop = enhance_for_handwriting(crop)
                        
                        buf = io.BytesIO()
                        enhanced_crop.save(buf, format="PNG")
                        vsmall = call_vision_doc(buf.getvalue())
                        
                        if vsmall.get("textAnnotations"):
                            raw = vsmall["textAnnotations"][0].get("description", "")
                            block_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                            lines.extend(block_lines)
    except Exception as e:
        logger.error(f"Error in block crop re-OCR: {e}")
    
    return lines

# ---------- Cache functions ----------
def cache_get(rid: int) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None

def cache_put(row: dict) -> None:
    if not supabase:
        return
    try:
        supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
    except Exception as e:
        logger.error(f"Cache put error: {e}")

# ---------- Rate limiting ----------
class TokenBucket:
    def __init__(self, rate_per_minute=60, capacity=None):
        self.rate = rate_per_minute/60.0
        self.capacity = capacity or rate_per_minute
        self.tokens = self.capacity
        self.last = time.time()
    
    def acquire(self, n=1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
    
    def wait(self, n=1):
        while not self.acquire(n):
            time.sleep(0.05)

_bucket = TokenBucket(60)

def limit_discogs(fn):
    def wrap(*a, **k):
        _bucket.wait(1)
        return fn(*a, **k)
    return wrap

# ---------- Discogs API functions ----------
@limit_discogs
def fetch_discogs_release_json(rid: int) -> Optional[dict]:
    try:
        headers = DGS_UA.copy()
        token = os.environ.get("DISCOGS_TOKEN")
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        
        r = requests.get(f"{DGS_API}/releases/{rid}", headers=headers, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.error(f"Discogs fetch error: {e}")
        return None

@limit_discogs
def discogs_search(params: Dict[str, str]) -> List[IdentifyCandidate]:
    p = params.copy()
    p.setdefault("type", "release")
    
    tok = os.environ.get("DISCOGS_TOKEN")
    if tok:
        p["token"] = tok
    
    headers = DGS_UA.copy()
    if tok:
        headers["Authorization"] = f"Discogs token={tok}"
    
    out = []
    try:
        r = requests.get(f"{DGS_API}/database/search", params=p, headers=headers, timeout=20)
        if r.status_code == 200:
            results = r.json().get("results", [])
            for it in results[:8]:
                url = it.get("resource_url", "")
                if "/releases/" not in url:
                    continue
                
                try:
                    rid = int(url.rstrip("/").split("/")[-1])
                except:
                    continue
                
                title_full = it.get("title", "")
                artist = None
                title = title_full
                
                if " - " in title_full:
                    parts = title_full.split(" - ", 1)
                    artist = parts[0].strip()
                    title = parts[1].strip()
                
                out.append(IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=artist,
                    title=title,
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.70
                ))
    except Exception as e:
        logger.error(f"Discogs search error: {e}")
    
    return out

# ---------- Main identify route ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        logger.info(f"Processing file: {file.filename}")
        image_bytes = await file.read()
        
        # Vision API calls
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        # Enhanced OCR processing
        text = handwriting_merge(image_bytes, text)
        extra = block_crop_reocr(image_bytes)
        
        if extra:
            merged = [ln.strip() for ln in (text[0].get("description", "").splitlines() if text else []) if ln.strip()]
            merged.extend(extra)
            text = [{"description": "\n".join(dict.fromkeys(merged))}]

        # Web detection path
        release_id, master_id, discogs_url = parse_discogs_web(web)
        candidates: List[IdentifyCandidate] = []

        if release_id:
            logger.info(f"Found web match: release_id={release_id}")
            cached = cache_get(release_id)
            if cached:
                candidates.append(IdentifyCandidate(
                    source="web_cache",
                    release_id=release_id,
                    discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"),
                    title=cached.get("title"),
                    label=cached.get("label"),
                    year=cached.get("year"),
                    cover_url=cached.get("cover_url"),
                    score=0.95
                ))
            else:
                rel = fetch_discogs_release_json(release_id)
                if rel:
                    row = {
                        "release_id": release_id,
                        "discogs_url": discogs_url or rel.get("uri") or f"https://www.discogs.com/release/{release_id}",
                        "artist": ", ".join(a.get("name", "") for a in rel.get("artists", [])),
                        "title": rel.get("title"),
                        "label": ", ".join(l.get("name", "") for l in rel.get("labels", [])),
                        "year": str(rel.get("year") or ""),
                        "cover_url": rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri", ""),
                        "payload": rel,
                    }
                    cache_put(row)
                    candidates.append(IdentifyCandidate(
                        source="web_live",
                        release_id=release_id,
                        discogs_url=row["discogs_url"],
                        artist=row["artist"],
                        title=row["title"],
                        label=row["label"],
                        year=row["year"],
                        cover_url=row["cover_url"],
                        score=0.90
                    ))

        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — select pressing",
                score=0.60
            ))

        # Perfect matching pipeline
        if not candidates:
            logger.info("No web matches, trying perfect matching pipeline")
            clean = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in ocr_lines(text) if ln.strip()]
            basic_candidates = improved_ocr_fallback_with_ml(text, clean, image_bytes)
            candidates = perfect_matching_pipeline(image_bytes, basic_candidates)

        logger.info(f"Returning {len(candidates)} candidates")
        return IdentifyResponse(candidates=candidates[:5])
        
    except Exception as exc:
        logger.error(f"Identify error: {exc}")
        raise HTTPException(500, str(exc))

# ---------- Debug endpoint ----------
@router.post("/api/debug-identify")
async def debug_identify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        web_urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
            urls = [item.get("url") for item in web.get(key, []) if item.get("url")]
            if urls:
                web_urls.extend(urls[:3])
        
        release_id, master_id, discogs_url = parse_discogs_web(web)
        
        raw_ocr = text[0].get("description", "") if text else ""
        
        text_enhanced = handwriting_merge(image_bytes, text)
        enhanced_ocr = text_enhanced[0].get("description", "") if text_enhanced else ""
        
        block_lines = block_crop_reocr(image_bytes)
        
        final_lines = ocr_lines(text_enhanced)
        if block_lines:
            merged = list(dict.fromkeys([*final_lines, *block_lines]))
            final_lines = merged
        
        clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in final_lines if ln.strip()]
        
        # Perfect matching debug info
        multi_ocr_results = multi_ocr_extraction(image_bytes)
        consensus_lines = consensus_ocr(multi_ocr_results)
        advanced_metadata = extract_advanced_metadata(consensus_lines)
        visual_characteristics = detect_label_shape(image_bytes)
        estimated_year = estimate_release_era(consensus_lines, image_bytes)
        
        test_queries = [
            {"q": "tron 001"},
            {"q": "tron revolta"},
            {"q": "unknown artist tron"},
        ]
        
        if advanced_metadata.get('label') and advanced_metadata.get('catno'):
            test_queries.append({"label": advanced_metadata['label'], "catno": advanced_metadata['catno']})
        
        search_results = {}
        for i, query in enumerate(test_queries):
            try:
                results = discogs_search(query)
                search_results[f"query_{i}"] = {
                    "query": query,
                    "count": len(results),
                    "results": [{"artist": r.artist, "title": r.title, "url": r.discogs_url} for r in results[:3]]
                }
            except Exception as e:
                search_results[f"query_{i}"] = {"query": query, "error": str(e)}
        
        return {
            "raw_ocr": raw_ocr,
            "enhanced_ocr": enhanced_ocr,
            "block_reocr": block_lines,
            "final_lines": final_lines,
            "cleaned_lines": clean_lines,
            "multi_ocr_results": multi_ocr_results,
            "consensus_lines": consensus_lines,
            "advanced_metadata": advanced_metadata,
            "visual_characteristics": visual_characteristics,
            "estimated_year": estimated_year,
            "web_detection": {
                "release_id": release_id,
                "master_id": master_id,
                "discogs_url": discogs_url,
                "sample_urls": web_urls
            },
            "search_results": search_results,
            "system_info": {
                "vision_key_set": bool(VISION_KEY),
                "discogs_token_set": bool(os.environ.get("DISCOGS_TOKEN")),
                "supabase_available": supabase is not None,
                "clip_available": CLIP_AVAILABLE,
                "cv2_available": CV2_AVAILABLE,
                "tesseract_available": TESSERACT_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
                "easyocr_available": EASYOCR_AVAILABLE,
            }
        }
        
    except Exception as exc:
        logger.error(f"Debug identify error: {exc}")
        return {"error": str(exc)}
