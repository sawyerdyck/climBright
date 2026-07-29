import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


SYSTEM_PROMPT = """
You are an indoor rock climbing coach.

You are given:

1) A wall photo

2) A JSON list of holds with reliable bounding boxes and reliable hold types.

Rules:

- Treat the JSON as the only reliable source for what holds exist and what type each hold is.

- Do NOT invent holds, hold types, or exact measurements.

- You may use the photo only for high-level context (specific color of route, orientation of hold, size of hold).

- Provide suggestions, not guarantees. Add a short safety disclaimer.

Task:

- Propose the most logical sequence from bottom to top using these holds.

- Explain how to climb each sequence in clear, step-by-step coaching language.

- Give a rough difficulty estimate (Easy / Moderate / Hard) and explain why using only hold types + spacing + route flow.

Output in JSON with keys: routeA, routeB, difficulty, notes which includes each hold that is involved in the sequence followed by the coordinated of that hold normalized to a single point as well as the size of its bounding box to allow the web ui to properly outline the hold.
"""

# --- Hold difficulty weights ---
HOLD_DIFFICULTY = {
    "crimp": 3,
    "pinch": 2.5,
    "sloper": 3,
    "pocket": 2.5,
    "undercling": 2,
    "sidepull": 1.5,
    "jug": 0.5,
    "volume": 0.5,
    "edge": 1.5,
    "horn": 1,
}

# Max normalized distance a climber can reach between holds.
# ponytail: tuned for indoor walls; 0.25 is ~arm span on a 3m wall photo.
MAX_REACH = 0.30
# Prefer holds that go upward; penalize big lateral jumps.
LATERAL_PENALTY = 1.5


def hold_difficulty_score(hold_type: str) -> float:
    return HOLD_DIFFICULTY.get(hold_type.lower().strip(), 1.5)


def dist_norm(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two normalized [x, y] points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_route_greedy(holds: List[dict], max_reach: float, prefer_direct: bool = True) -> List[dict]:
    """
    Greedy bottom-to-top route builder.

    Starts from the lowest hold(s), picks the best next hold going upward
    within reach distance, preferring holds that are more directly above
    (less lateral movement) and easier to grab.

    If prefer_direct=False, allows bigger lateral moves (harder route).
    """
    if not holds:
        return []

    # Sort by Y descending (highest center_norm[1] = lowest on wall in image coords,
    # BUT normalized: y=0 is top of image, y=1 is bottom).
    # So y=1 is the bottom of the wall. Start from highest y (bottom).
    sorted_holds = sorted(holds, key=lambda h: h["center_norm"][1], reverse=True)

    # Start from the lowest hold (highest y)
    start = sorted_holds[0]
    route = [start]
    used = {start["id"]}

    current = start
    max_steps = len(holds)  # safety cap

    for _ in range(max_steps):
        cx, cy = current["center_norm"]

        # Find candidates: above current (lower y), within reach
        candidates = []
        for h in holds:
            if h["id"] in used:
                continue
            hx, hy = h["center_norm"]
            # Must go upward (lower y value)
            if hy >= cy:
                continue

            d = dist_norm([cx, cy], [hx, hy])
            if d > max_reach:
                continue

            # Score: prefer close, prefer directly above, prefer easier holds for routeA
            vertical_gain = cy - hy  # positive = going up
            lateral_shift = abs(hx - cx)

            # ponytail: simple scoring — vertical gain good, lateral bad, distance matters
            score = vertical_gain - (lateral_shift * LATERAL_PENALTY if prefer_direct else lateral_shift * 0.5) - d * 0.5

            candidates.append((score, h))

        if not candidates:
            break

        # Pick best candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        route.append(best)
        used.add(best["id"])
        current = best

    # Route is built bottom-to-top: step 1 = bottom (start), last step = top (finish)
    return route


def build_route_hard(holds: List[dict], max_reach: float) -> List[dict]:
    """
    Builds a harder route variant: bigger moves, allows more lateral movement,
    skips easier holds when possible.
    """
    if len(holds) <= 3:
        return build_route_greedy(holds, max_reach, prefer_direct=False)

    # Filter to prefer harder hold types; fall back to all if too few remain
    hard_types = {"crimp", "pinch", "sloper", "pocket", "undercling", "sidepull"}
    hard_holds = [h for h in holds if (h.get("type") or "").lower() in hard_types]

    # Need at least a start and end hold from the full set
    sorted_all = sorted(holds, key=lambda h: h["center_norm"][1], reverse=True)
    bottom = sorted_all[0]
    top = sorted_all[-1]

    pool = hard_holds if len(hard_holds) >= 3 else holds
    # Ensure start/end are in pool
    pool_ids = {h["id"] for h in pool}
    if bottom["id"] not in pool_ids:
        pool = [bottom] + pool
    if top["id"] not in pool_ids:
        pool = pool + [top]

    return build_route_greedy(pool, max_reach * 1.2, prefer_direct=False)


def score_difficulty(route: List[dict], all_holds: List[dict]) -> str:
    """Score route difficulty based on hold types, spacing, and lateral variance."""
    if len(route) < 2:
        return "Easy"

    # Hold type difficulty
    type_scores = [hold_difficulty_score(h.get("type", "Unknown")) for h in route]
    avg_type = sum(type_scores) / len(type_scores)

    # Average gap between consecutive holds
    gaps = []
    lateral_shifts = []
    for i in range(len(route) - 1):
        a = route[i]["center_norm"]
        b = route[i + 1]["center_norm"]
        gaps.append(dist_norm(a, b))
        lateral_shifts.append(abs(a[0] - b[0]))

    avg_gap = sum(gaps) / len(gaps)
    avg_lateral = sum(lateral_shifts) / len(lateral_shifts) if lateral_shifts else 0

    # Composite score
    score = avg_type * 2 + avg_gap * 10 + avg_lateral * 5

    if score >= 8:
        return "Hard"
    elif score >= 4.5:
        return "Moderate"
    else:
        return "Easy"


def format_route_output(route: List[dict]) -> List[dict]:
    """Format route holds for JSON output."""
    return [
        {
            "id": h["id"],
            "type": h.get("type", "Unknown"),
            "center_norm": h["center_norm"],
            "bbox_wh_norm": h["bbox_wh_norm"],
        }
        for h in route
    ]


def build_local_coach(normalized: dict) -> dict:
    holds = normalized.get("holds", [])
    if not holds:
        return {
            "routeA": [],
            "routeB": [],
            "difficulty": "Easy",
            "notes": "No holds provided; unable to generate a route.",
        }

    # Route A: standard greedy route (prefer direct upward movement)
    routeA = build_route_greedy(holds, MAX_REACH, prefer_direct=True)

    # Route B: harder variant (bigger moves, harder holds)
    routeB = build_route_hard(holds, MAX_REACH)

    # If routeB ended up identical to routeA, just skip every other hold
    if [h["id"] for h in routeB] == [h["id"] for h in routeA] and len(routeA) > 3:
        routeB = routeA[::2]
        if routeB[-1]["id"] != routeA[-1]["id"]:
            routeB.append(routeA[-1])

    difficulty = score_difficulty(routeA, holds)

    notes_parts = [
        "Local coach (offline). Route generated using reach-constrained greedy pathfinding.",
        f"Route A: {len(routeA)} moves (standard). Route B: {len(routeB)} moves (harder variant).",
        "Verify on the wall before climbing. Use proper safety equipment.",
    ]

    return {
        "routeA": format_route_output(routeA),
        "routeB": format_route_output(routeB),
        "difficulty": difficulty,
        "notes": " ".join(notes_parts),
    }


def load_files(image_path: str, json_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing json: {json_path}")

    img = Image.open(image_path).convert("RGB")
    with open(json_path, "r", encoding="utf-8") as f:
        hold_data = json.load(f)
    return img, hold_data


def normalize_holds(hold_data: dict, img_w: int, img_h: int) -> dict:
    """
    Adds center_norm and bbox_wh_norm to each hold using bbox [x1,y1,x2,y2].
    Accepts top-level "holds" or "objects".
    """
    key = "holds" if "holds" in hold_data else ("objects" if "objects" in hold_data else None)
    if key is None:
        raise ValueError('JSON must have top-level key "holds" or "objects".')

    out = {"image_size": {"w": img_w, "h": img_h}, "holds": []}

    for i, h in enumerate(hold_data[key]):
        bbox = h.get("bbox") or h.get("box")
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Hold index {i} missing bbox/box: expected [x1,y1,x2,y2]")

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(1.0, (x2 - x1))
        bh = max(1.0, (y2 - y1))

        hold_id = h.get("id", i)
        hold_type = h.get("type", h.get("class_name", h.get("label", "Unknown")))

        out["holds"].append({
            "id": hold_id,
            "type": hold_type,
            "bbox": [x1, y1, x2, y2],
            "center_norm": [cx / img_w, cy / img_h],
            "bbox_wh_norm": [bw / img_w, bh / img_h],
        })

    return out


def generate_gemini_coach(img: Image.Image, normalized: dict, model: str) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        genai_client = genai.Client(api_key=api_key)
    except Exception:
        try:
            from google.genai import client, types  # type: ignore

            genai_client = client.Client(api_key=api_key)
        except Exception as e:
            sys.stderr.write(f"Gemini import/init failed; falling back to local coach. Error: {e}\n")
            return None

    holds_json_str = json.dumps(normalized, ensure_ascii=False)

    try:
        response = genai_client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
            contents=[
                img,
                "Here is the hold data JSON (this is the only reliable hold info):",
                holds_json_str,
            ],
        )
        text = getattr(response, "text", None) or ""
        return response.json()
    except Exception as e:
        sys.stderr.write(f"Gemini request failed; falling back to local coach. Error: {e}\n")
        return None


def run() -> None:
    parser = argparse.ArgumentParser(description="Generate climbing routes from holds JSON (Gemini or local fallback).")
    parser.add_argument("--image", required=True, help="Path to wall image")
    parser.add_argument("--json", required=True, help="Path to holds JSON (must include top-level 'holds')")
    parser.add_argument("--model", default="models/gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--local", action="store_true", help="Force local coach (skip Gemini)")
    args = parser.parse_args()

    img, hold_data = load_files(args.image, args.json)
    img_w, img_h = img.size
    normalized = normalize_holds(hold_data, img_w, img_h)

    result: Optional[Dict[str, Any]] = None
    if not args.local:
        result = generate_gemini_coach(img, normalized, model=args.model)
    if result is None:
        result = build_local_coach(normalized)

    # IMPORTANT: stdout must be JSON-only for the Node server parser
    print(json.dumps(result))

    return json.dumps(result)


if __name__ == "__main__":
    run()
