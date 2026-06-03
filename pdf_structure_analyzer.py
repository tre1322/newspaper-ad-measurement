#!/usr/bin/env python3
"""PDF structure extraction for ad candidate generation.

This module is a **candidate generator** only. It does NOT classify candidates
as ads or editorial — that's Claude Vision's job (see ad_judge.py). All
previous rule-based scoring (business keywords, editorial patterns, house-ad
phrases) has been removed. What remains is infrastructure:

  1. Parse PDF pages via PyMuPDF into text blocks, images, drawings.
  2. Emit candidates from two geometric signals:
     - Bordered regions: any drawing whose bounding box is ad-sized.
     - Content clusters: graph-based spatial clusters of text blocks + images
       (catches ads that have no drawn border, or whose border is broken into
       multiple path components).

The orchestrator in app.py sends each candidate crop to Claude for the
AD/EDITORIAL/FURNITURE verdict.
"""

import fitz  # PyMuPDF
import os
from typing import List, Dict, Optional


class PDFStructureAdDetector:
    """Extract geometric ad candidates from PDF structure.

    Public entry point: `extract_candidates(pdf_path, page_number)`.

    Returns a dict:
        {
            'bordered': [<candidate>, ...],
            'clusters': [<candidate>, ...],
            'images':   [<candidate>, ...],
            'page_rect': fitz.Rect,
        }

    Each <candidate> is:
        {
            'x': float,   # PDF point space (0,0 top-left)
            'y': float,
            'width': float,
            'height': float,
            'source': 'structure_bordered' | 'structure_cluster' | 'structure_image',
            'text_preview': str,   # first ~200 chars of interior text, for profile learning
        }
    """

    # ---------- Public entry ------------------------------------------------

    @classmethod
    def detect_ads_from_pdf_structure(cls, pdf_path, page_number, publication_type=None):
        """Deprecated shim. Returns []. Real detection goes through the
        LLM-judge pipeline in app.py via extract_candidates()."""
        return []

    @classmethod
    def extract_candidates(cls, pdf_path: str, page_number: int) -> Dict:
        if not os.path.exists(pdf_path):
            print(f"[pdf_structure] PDF not found: {pdf_path}")
            return {'bordered': [], 'clusters': [], 'page_rect': None}
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[pdf_structure] failed to open PDF: {e}")
            return {'bordered': [], 'clusters': [], 'page_rect': None}
        try:
            if page_number < 1 or page_number > len(doc):
                return {'bordered': [], 'clusters': [], 'page_rect': None}
            page = doc[page_number - 1]
            structure = cls._extract_page_structure(page)
            bordered = cls._find_bordered_candidates(structure)
            clusters = cls._find_cluster_candidates(structure)
            images = cls._find_image_candidates(structure)
            return {
                'bordered': bordered,
                'clusters': clusters,
                'images': images,
                'page_rect': structure['page_rect'],
            }
        finally:
            doc.close()

    # ---------- Extraction --------------------------------------------------

    @classmethod
    def _extract_page_structure(cls, page) -> Dict:
        """Pull text blocks, image placements, and vector drawings out of the page."""
        structure = {
            'text_blocks': [],
            'images': [],
            'drawings': [],
            'page_rect': page.rect,
        }

        # Text blocks
        try:
            text_dict = page.get_text("dict")
            for i, block in enumerate(text_dict.get("blocks", [])):
                if "lines" not in block:
                    continue  # image block, handled separately
                info = cls._summarize_text_block(block, i)
                if info:
                    structure['text_blocks'].append(info)
        except Exception as e:
            print(f"[pdf_structure] text extraction failed: {e}")

        # Images
        try:
            for i, img in enumerate(page.get_images()):
                try:
                    for j, rect in enumerate(page.get_image_rects(img)):
                        if rect.width <= 0 or rect.height <= 0:
                            continue
                        structure['images'].append({
                            'id': f"img_{i}_{j}",
                            'bounds': [float(rect.x0), float(rect.y0),
                                       float(rect.x1), float(rect.y1)],
                            'width': float(rect.width),
                            'height': float(rect.height),
                            'area': float(rect.width * rect.height),
                        })
                except Exception as e:
                    print(f"[pdf_structure]   image {i} failed: {e}")
        except Exception as e:
            print(f"[pdf_structure] image extraction failed: {e}")

        # Vector drawings
        try:
            for i, drawing in enumerate(page.get_drawings()):
                try:
                    rect = drawing.get('rect')
                    if rect is None or rect.width <= 0 or rect.height <= 0:
                        continue
                    structure['drawings'].append({
                        'id': f"drawing_{i}",
                        'bounds': [float(rect.x0), float(rect.y0),
                                   float(rect.x1), float(rect.y1)],
                        'width': float(rect.width),
                        'height': float(rect.height),
                        'area': float(rect.width * rect.height),
                        'item_count': len(drawing.get('items', [])),
                    })
                except Exception as e:
                    print(f"[pdf_structure]   drawing {i} failed: {e}")
        except Exception as e:
            print(f"[pdf_structure] drawing extraction failed: {e}")

        return structure

    @classmethod
    def _summarize_text_block(cls, block, block_id) -> Optional[Dict]:
        bbox = block.get("bbox")
        if not bbox:
            return None
        text_parts = []
        total_chars = 0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                text_parts.append(t)
                total_chars += len(t)
        text = " ".join(text_parts).strip()
        if not text or total_chars < 3:
            return None
        return {
            'id': f"text_{block_id}",
            'bounds': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            'width': float(bbox[2] - bbox[0]),
            'height': float(bbox[3] - bbox[1]),
            'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            'text_content': text,
            'char_count': total_chars,
        }

    # ---------- Bordered candidates ----------------------------------------

    # Geometric filters for candidate bounding boxes (all in PDF points).
    # Deliberately LOOSE / high-recall: Claude Vision is the precision gate, so a
    # candidate that isn't really an ad gets classified EDITORIAL/FURNITURE and
    # dropped downstream. Missing a candidate here is unrecoverable — it can never
    # be judged or marked — so tune toward over-emitting.
    MIN_WIDTH_PT = 54          # ~0.75 inch (catch narrow single-column ads)
    MIN_HEIGHT_PT = 30         # ~0.4 inch
    MIN_AREA_PT2 = 3_000       # catch small classified-line / service ads
    MAX_AREA_PT2 = 1_600_000   # absolute fallback when page area is unknown
    MAX_AREA_FRAC = 0.95       # else allow up to 95% of the page (full-page ads),
                               # excluding only the whole-page border itself
    MAX_ASPECT = 16.0          # allow tall single-column / skinny banner ads

    @staticmethod
    def _page_area(structure: Dict):
        pr = structure.get('page_rect')
        if pr is None:
            return None
        try:
            return float(pr.width) * float(pr.height)
        except Exception:
            return None

    @classmethod
    def _find_bordered_candidates(cls, structure: Dict) -> List[Dict]:
        """Any vector drawing whose bounding rect is ad-sized."""
        page_area = cls._page_area(structure)
        out: List[Dict] = []
        for d in structure['drawings']:
            if not cls._is_ad_sized(d['width'], d['height'], d['area'], page_area):
                continue
            text_preview = cls._text_inside(d['bounds'], structure['text_blocks'], limit=200)
            out.append({
                'x': d['bounds'][0],
                'y': d['bounds'][1],
                'width': d['width'],
                'height': d['height'],
                'source': 'structure_bordered',
                'text_preview': text_preview,
            })
        return out

    @classmethod
    def _is_ad_sized(cls, w: float, h: float, area: float, page_area: float = None) -> bool:
        if w < cls.MIN_WIDTH_PT or h < cls.MIN_HEIGHT_PT:
            return False
        if area < cls.MIN_AREA_PT2:
            return False
        # Cap relative to the page so near-full-page ads pass but the whole-page
        # border (~100% of page area) is excluded. Fall back to a fixed ceiling
        # when page dimensions are unknown.
        max_area = cls.MAX_AREA_FRAC * page_area if page_area else cls.MAX_AREA_PT2
        if area > max_area:
            return False
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > cls.MAX_ASPECT:
            return False
        return True

    # ---------- Standalone image candidates --------------------------------

    @classmethod
    def _find_image_candidates(cls, structure: Dict) -> List[Dict]:
        """Each ad-sized raster image becomes its own candidate.

        Catches full-bleed / borderless photo ads that the cluster detector
        misses (a single image with no neighboring text is one element, below
        CLUSTER_MIN_ELEMENTS). Overlap with bordered/cluster candidates is
        resolved by the area-descending dedupe in app.py (the outer frame wins).
        """
        page_area = cls._page_area(structure)
        out: List[Dict] = []
        for im in structure['images']:
            if not cls._is_ad_sized(im['width'], im['height'], im['area'], page_area):
                continue
            text_preview = cls._text_inside(im['bounds'], structure['text_blocks'], limit=200)
            out.append({
                'x': im['bounds'][0],
                'y': im['bounds'][1],
                'width': im['width'],
                'height': im['height'],
                'source': 'structure_image',
                'text_preview': text_preview,
            })
        return out

    # ---------- Cluster candidates -----------------------------------------

    CLUSTER_MERGE_GAP_PT = 32.0   # elements within this distance merge into the same cluster
    CLUSTER_MIN_ELEMENTS = 2      # minimum elements in a cluster (text blocks + images)

    @classmethod
    def _find_cluster_candidates(cls, structure: Dict) -> List[Dict]:
        """Spatial clusters of text + images that form coherent page regions.

        Catches ads with no drawn border (or borders drawn as disconnected
        lines / raster). Uses union-find on bounding-box proximity.
        """
        page_area = cls._page_area(structure)
        elements: List[Dict] = []
        for t in structure['text_blocks']:
            elements.append({'bounds': t['bounds'], 'is_image': False, 'text': t.get('text_content', '')})
        for i in structure['images']:
            elements.append({'bounds': i['bounds'], 'is_image': True, 'text': ''})

        n = len(elements)
        if n < cls.CLUSTER_MIN_ELEMENTS:
            return []

        # Union-find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        gap = cls.CLUSTER_MERGE_GAP_PT
        for i in range(n):
            bi = elements[i]['bounds']
            for j in range(i + 1, n):
                bj = elements[j]['bounds']
                if cls._bbox_gap(bi, bj) <= gap:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        out: List[Dict] = []
        for members in groups.values():
            if len(members) < cls.CLUSTER_MIN_ELEMENTS:
                continue
            # Combined bounding box
            xs_min = min(elements[m]['bounds'][0] for m in members)
            ys_min = min(elements[m]['bounds'][1] for m in members)
            xs_max = max(elements[m]['bounds'][2] for m in members)
            ys_max = max(elements[m]['bounds'][3] for m in members)
            w = xs_max - xs_min
            h = ys_max - ys_min
            area = w * h
            if not cls._is_ad_sized(w, h, area, page_area):
                continue
            text_preview = " ".join(elements[m]['text'] for m in members if elements[m]['text'])[:200]
            out.append({
                'x': xs_min,
                'y': ys_min,
                'width': w,
                'height': h,
                'source': 'structure_cluster',
                'text_preview': text_preview.strip(),
            })
        return out

    @staticmethod
    def _bbox_gap(a, b) -> float:
        """Minimum edge-to-edge distance between two axis-aligned boxes; 0 if overlapping."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0.0, max(by1 - ay2, ay1 - by2))
        if dx == 0 and dy == 0:
            return 0.0
        return (dx * dx + dy * dy) ** 0.5

    # ---------- Interior text helpers --------------------------------------

    @staticmethod
    def _text_inside(border_bounds, text_blocks, limit: int = 200) -> str:
        bx1, by1, bx2, by2 = border_bounds
        parts: List[str] = []
        for t in text_blocks:
            tx1, ty1, tx2, ty2 = t['bounds']
            if (bx1 - 2 <= tx1 and tx2 <= bx2 + 2 and
                    by1 - 2 <= ty1 and ty2 <= by2 + 2):
                parts.append(t.get('text_content', ''))
        combined = " ".join(parts).strip()
        return combined[:limit]
