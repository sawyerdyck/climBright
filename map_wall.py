import argparse
import json
import math
import os
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


def load_holds(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict) and "holds" in data:
        holds_source = data["holds"]
    else:
        holds_source = data

    holds = []
    for idx, item in enumerate(holds_source):
        hold_id = int(item.get("id", idx))
        cx = item.get("cx")
        cy = item.get("cy")
        box = item.get("box") or item.get("bbox")

        if cx is None or cy is None:
            if not box or len(box) != 4:
                raise ValueError(f"Hold {hold_id} missing center and box data")
            x1, y1, x2, y2 = map(float, box)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

        class_id = int(item.get("class_id", item.get("type", 0)))
        hold = {
            "id": hold_id,
            "cx": float(cx),
            "cy": float(cy),
            "type": max(0, class_id),
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
            "box": box,
        }
        holds.append(hold)

    return holds


def attach_wall_coordinates(holds, H):
    centers_px = [(h["cx"], h["cy"]) for h in holds]
    centers_wall = pixel_to_wall(H, centers_px)
    for hold, (X, Y) in zip(holds, centers_wall):
        hold["X"] = float(X)
        hold["Y"] = float(Y)


def select_terminal_holds(holds, start_band=0.1, finish_band=0.1):
    ys = [h["Y"] for h in holds]
    max_y = max(ys)
    min_y = min(ys)

    start_threshold = max_y - start_band
    finish_threshold = min_y + finish_band

    start_ids = [h["id"] for h in holds if h["Y"] >= start_threshold]
    finish_ids = [h["id"] for h in holds if h["Y"] <= finish_threshold]

    if not start_ids:
        start_ids = [max(holds, key=lambda h: h["Y"])["id"]]
    if not finish_ids:
        finish_ids = [min(holds, key=lambda h: h["Y"])["id"]]

    return start_ids, finish_ids


def build_edge_lookup(adj):
    table = {}
    for src, edges in adj.items():
        table[src] = {dst: cost for dst, cost in edges}
    return table


def path_cost(path, edge_lookup):
    total = 0.0
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        total += edge_lookup.get(src, {}).get(dst, 0.0)
    return total


def enumerate_paths(adj, start_ids, finish_ids, max_hops=12, max_paths=200):
    finish_set = set(finish_ids)
    collected = []

    for start in start_ids:
        stack = [(start, [start])]
        while stack and len(collected) < max_paths:
            node, path = stack.pop()
            if node in finish_set and len(path) > 1:
                collected.append(path)
                continue

            if len(path) >= max_hops:
                continue

            for neighbor, _ in adj.get(node, []):
                if neighbor in path:
                    continue
                next_path = path + [neighbor]
                stack.append((neighbor, next_path))

    return collected


def draw_overlay(image_path, holds, adj, highlight_paths, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    canvas = img.copy()
    hold_map = {h["id"]: h for h in holds}

    # Draw all feasible moves first
    for src_id, edges in adj.items():
        src = hold_map[src_id]
        src_pt = (int(round(src["cx"])), int(round(src["cy"])))
        for dst_id, _ in edges:
            dst = hold_map[dst_id]
            dst_pt = (int(round(dst["cx"])), int(round(dst["cy"])))
            cv2.line(canvas, src_pt, dst_pt, (0, 255, 255), 1, cv2.LINE_AA)

    colors = [
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]
    for idx, path in enumerate(highlight_paths):
        color = colors[idx % len(colors)]
        for i in range(len(path) - 1):
            a = hold_map[path[i]]
            b = hold_map[path[i + 1]]
            a_pt = (int(round(a["cx"])), int(round(a["cy"])))
            b_pt = (int(round(b["cx"])), int(round(b["cy"])))
            cv2.line(canvas, a_pt, b_pt, color, 3, cv2.LINE_AA)

    for hold in holds:
        center = (int(round(hold["cx"])), int(round(hold["cy"])))
        cv2.circle(canvas, center, 6, (0, 0, 0), -1)
        cv2.circle(canvas, center, 4, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            str(hold["id"]),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, canvas)

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
def main():
    parser = argparse.ArgumentParser(description="Generate climbing route graph from detected holds")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the wall image")
    parser.add_argument("--holds", type=str, required=True, help="JSON file containing detected holds")
    parser.add_argument("--homography", type=str, help="Existing homography .npy file to reuse")
    parser.add_argument("--save-homography", type=str, dest="save_path", help="Optional output path for homography .npy")
    parser.add_argument("--reach", type=float, default=0.18, help="Maximum normalized reach between holds")
    parser.add_argument("--max-down", type=float, default=0.03, help="Allowed downward move in normalized units")
    parser.add_argument("--max-side", type=float, default=0.25, help="Allowed sideways move in normalized units")
    parser.add_argument("--start-band", type=float, default=0.12, help="Lower band (normalized) treated as potential starts")
    parser.add_argument("--finish-band", type=float, default=0.12, help="Upper band (normalized) treated as potential finishes")
    parser.add_argument("--max-hops", type=int, default=12, help="Maximum holds per enumerated path")
    parser.add_argument("--max-paths", type=int, default=300, help="Maximum enumerated paths to retain")
    parser.add_argument("--highlight-top", type=int, default=5, help="How many best paths to highlight on overlay")
    parser.add_argument("--overlay-out", type=str, help="Path to write overlay image with routes")
    parser.add_argument("--paths-out", type=str, help="Path to write JSON summary of computed paths")
    args = parser.parse_args()

    holds = load_holds(args.holds)
    if not holds:
        raise ValueError("No holds loaded from input file")

    if args.homography:
        H = np.load(args.homography)
        print(f"Loaded homography from {args.homography}")
    else:
        corners = click_4_corners(args.image)
        print("Corners (TL, TR, BR, BL):", corners)
        H = compute_homography(corners)
        print("Homography matrix:\n", H)
        if args.save_path:
            np.save(args.save_path, H)
            print(f"Saved homography to {args.save_path}")

    attach_wall_coordinates(holds, H)

    adj = build_graph(
        holds,
        reach=args.reach,
        max_down=args.max_down,
        max_side=args.max_side,
    )

    start_ids, finish_ids = select_terminal_holds(
        holds,
        start_band=args.start_band,
        finish_band=args.finish_band,
    )

    print(f"Start holds: {start_ids}")
    print(f"Finish holds: {finish_ids}")

    edge_lookup = build_edge_lookup(adj)
    enumerated = enumerate_paths(
        adj,
        start_ids,
        finish_ids,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
    )

    path_records = []
    for path in enumerated:
        path_records.append(
            {
                "sequence": path,
                "cost": path_cost(path, edge_lookup),
                "length": len(path),
            }
        )

    path_records.sort(key=lambda item: (item["cost"], item["length"]))

    shortest_map = {}
    for start in start_ids:
        for finish in finish_ids:
            route = dijkstra(adj, start, finish)
            if not route:
                continue
            key = tuple(route)
            cost = path_cost(route, edge_lookup)
            best_cost = shortest_map.get(key)
            if best_cost is None or cost < best_cost:
                shortest_map[key] = cost

    shortest_list = [
        {"sequence": list(seq), "cost": cost, "length": len(seq)}
        for seq, cost in shortest_map.items()
    ]
    shortest_list.sort(key=lambda item: (item["cost"], item["length"]))

    print(f"Enumerated {len(path_records)} paths (max {args.max_paths})")
    print(f"Unique shortest paths: {len(shortest_list)}")

    if args.paths_out:
        paths_payload = {
            "image": args.image,
            "holds_file": args.holds,
            "homography": args.homography or args.save_path,
            "start_ids": start_ids,
            "finish_ids": finish_ids,
            "graph_params": {
                "reach": args.reach,
                "max_down": args.max_down,
                "max_side": args.max_side,
            },
            "enumerated_paths": path_records,
            "shortest_paths": shortest_list,
        }
        out_dir = os.path.dirname(args.paths_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.paths_out, "w", encoding="utf-8") as fp:
            json.dump(paths_payload, fp, indent=2)
        print(f"Wrote path summary to {args.paths_out}")

    highlights = [item["sequence"] for item in shortest_list[: args.highlight_top]]
    if not highlights:
        highlights = [item["sequence"] for item in path_records[: args.highlight_top]]

    if args.overlay_out:
        draw_overlay(args.image, holds, adj, highlights, args.overlay_out)
        print(f"Wrote overlay to {args.overlay_out}")


if __name__ == "__main__":
    main()

