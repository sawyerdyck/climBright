import numpy as np
import cv2

def compute_homography(corners_px):
    """
    corners_px: list of 4 (x,y) points in this order:
      top-left, top-right, bottom-right, bottom-left
    returns: 3x3 homography matrix H mapping pixel -> normalized wall coords
    """
    src = np.array(corners_px, dtype=np.float32)

    # normalized rectangle corners
    dst = np.array([
        [0.0, 0.0],  # top-left
        [1.0, 0.0],  # top-right
        [1.0, 1.0],  # bottom-right
        [0.0, 1.0],  # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H

def pixel_to_wall(H, points_px):
    """
    points_px: Nx2 array/list of pixel points
    returns: Nx2 array of (X,Y) in [0,1]x[0,1]
    """
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return warped
import cv2

def click_4_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"Clicked: {(x,y)}  ({len(pts)}/4)")
            cv2.circle(img, (x,y), 6, (0,255,0), -1)
            cv2.imshow("click corners", img)

    cv2.imshow("click corners", img)
    cv2.setMouseCallback("click corners", on_mouse)

    print("Click 4 corners in order: TL, TR, BR, BL. Press ESC when done.")
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        if len(pts) == 4:
            break

    cv2.destroyAllWindows()
    if len(pts) != 4:
        raise RuntimeError("Need exactly 4 corner clicks.")
    return pts

corners = click_4_corners("wall.jpg")
H = compute_homography(corners)

hold_centers_px = [(cx, cy), (cx2, cy2), ...]
hold_centers_wall = pixel_to_wall(H, hold_centers_px)
dist = sqrt((Xj-Xi)^2 + (Yj-Yi)^2)
import math

def build_graph(holds, reach=0.18, max_down=0.03, max_side=0.25):
    """
    holds: list of dicts, each:
      {"id": i, "X":..., "Y":..., "type":...}
    reach: max move distance
    max_down: allow small downward moves (optional)
    max_side: limit huge sideways moves
    returns: adjacency dict id -> list of (neighbor_id, cost)
    """
    adj = {h["id"]: [] for h in holds}

    for a in holds:
        for b in holds:
            if a["id"] == b["id"]:
                continue

            dx = b["X"] - a["X"]
            dy = b["Y"] - a["Y"]

            # mostly go up (allow tiny down)
            if dy < -max_down:
                continue

            if abs(dx) > max_side:
                continue

            d = math.hypot(dx, dy)
            if d > reach:
                continue

            # cost: distance + (optional) difficulty penalty based on type
            type_penalty = type_cost(b["type"])
            cost = d + type_penalty

            adj[a["id"]].append((b["id"], cost))

    return adj

def type_cost(t):
    """
    You decide these numbers.
    If type 0 is easiest and 5 is hardest, penalize harder types.
    Start with small penalties; tune later.
    """
    penalties = [0.00, 0.03, 0.06, 0.09, 0.12, 0.15]
    return penalties[t] if 0 <= t < len(penalties) else 0.0
def pick_start_finish_auto(holds):
    start = max(holds, key=lambda h: h["Y"])  # bottom-most
    finish = min(holds, key=lambda h: h["Y"]) # top-most
    return start["id"], finish["id"]
import heapq

def dijkstra(adj, start_id, goal_id):
    pq = [(0.0, start_id)]
    dist = {start_id: 0.0}
    prev = {start_id: None}

    while pq:
        cost_u, u = heapq.heappop(pq)
        if u == goal_id:
            break
        if cost_u != dist.get(u, float("inf")):
            continue

        for v, w in adj.get(u, []):
            nd = cost_u + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal_id not in dist:
        return None  # no path found

    # reconstruct path
    path = []
    cur = goal_id
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path
# holds list example after detection+classification:
# holds = [{"id":0,"cx":123,"cy":456,"type":2}, ...]

# 1) user clicks wall corners
corners = click_4_corners("wall.jpg")
H = compute_homography(corners)

# 2) convert each hold center to wall coords
centers_px = [(h["cx"], h["cy"]) for h in holds]
centers_wall = pixel_to_wall(H, centers_px)

for h, (X, Y) in zip(holds, centers_wall):
    h["X"] = float(X)
    h["Y"] = float(Y)

# 3) build graph
adj = build_graph(holds, reach=0.18)

# 4) choose start/finish
start_id, finish_id = pick_start_finish_auto(holds)
# OR: user clicks start/finish holds instead

# 5) find path
path = dijkstra(adj, start_id, finish_id)
print("Suggested hold sequence:", path)

