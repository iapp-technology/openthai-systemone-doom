#!/usr/bin/env python3
"""OpenThai-SystemOne plays Doom on your machine.

No vision, no text generation: every step the game's symbolic state (health, ammo, enemies with distance/bearing, wall
depth, last actions) is sent as text to the decision model, which answers two typed questions in ONE forward pass:
  choice  -> which of the 7 actions to take          (the key that gets pressed)
  noul    -> is an enemy inside the crosshair         (threat bar)

Backends
  --backend iapp   (default) call the iApp API Gateway with your API key   -> register at https://iapp.co.th
  --backend url    any server that speaks the same POST /v1/systemone contract (e.g. your own uvicorn)
  --backend local  run the open weights on this machine (pip install openthai-systemone; CUDA / Apple MPS)

Examples
  export IAPP_API_KEY=iapp_live_xxxxxxxx
  python play_doom.py                                   # live HUD window (needs opencv-python) + Doom window
  python play_doom.py --scenario defend_the_center --lang th --record my_clip.mp4
  python play_doom.py --backend local --model iapp/OpenThai-SystemOne --tics 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IAPP_URL = "https://api.iapp.co.th/v3/store/openthai/systemone"

ACTIONS = {  # option name -> description (the model only ever sees these strings)
    "MOVE_FORWARD": "run forward",
    "TURN_LEFT": "turn the view left",
    "TURN_RIGHT": "turn the view right",
    "ATTACK": "fire the weapon at what is in the crosshair",
    "MOVE_LEFT": "strafe left without turning",
    "MOVE_RIGHT": "strafe right without turning",
    "MOVE_BACKWARD": "step back",
}
ENEMY_NAMES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "Spectre", "Cacodemon", "LostSoul", "HellKnight", "BaronOfHell", "Revenant", "Arachnotron", "Mancubus"}
POLICY = ("Rules: if an enemy is VISIBLE and its bearing is within 8 degrees, ATTACK. If a VISIBLE enemy is to the left (positive "
          "bearing) TURN_LEFT; to the right (negative bearing) TURN_RIGHT. If no enemy is visible, {goal}. If STUCK, turn or strafe.")
SCENARIOS = {  # scenario -> (goal, situation)
    "deadly_corridor": ("MOVE_FORWARD toward the green armor at the end of the corridor", "You control a Doom marine in a corridor; enemies shoot from alcoves on both sides; a green armor at the far end is the goal."),
    "defend_the_center": ("keep scanning by turning", "You stand in the middle of a circular arena; monsters approach from all directions; survive and kill as many as possible."),
    "defend_the_line": ("keep scanning by turning", "You stand at one end of a hall; monsters spawn at the far end and advance; kill them before they reach you."),
    "health_gathering": ("MOVE_FORWARD toward the nearest Medikit, turning toward it first", "You are losing health on acid floor; medikits restore it; walk over as many medikits as you can."),
    "basic": ("MOVE_LEFT or MOVE_RIGHT to line the monster up with the crosshair", "A single monster stands in front of you; strafe until it is centered, then fire."),
    "my_way_home": ("MOVE_FORWARD through rooms toward the green armor, turning when a wall is ahead", "You are in a maze of rooms; find the green armor."),
    "take_cover": ("MOVE_LEFT or MOVE_RIGHT to dodge fireballs, keep changing side", "Monsters throw fireballs at you from the far end; you cannot fight, only dodge sideways."),
}
THREAT_Q = "Is an enemy inside the crosshair (bearing within 6 degrees, in view) so that firing now would hit it?"
HUD = {
    "en": {"title": "OpenThai-SystemOne  ·  Doom", "sub": "0.8B decision model · no vision · no text generation · 1 pass per step",
           "state": "STATE (what the model reads)", "choice": "CHOICE · next action", "noul": "NOUL · enemy in crosshair?", "pyes": "P(yes)",
           "lat": "latency {lat:4.0f} ms  ·  {dps:3.1f} decisions/s  ·  step {step}", "stats": "health {hp:.0f}   kills {kills:.0f}   output tokens: 0",
           "foot": "huggingface.co/iapp/OpenThai-SystemOne · Apache-2.0"},
    "th": {"title": "OpenThai-SystemOne  ·  Doom", "sub": "โมเดลตัดสินใจ 0.8B · ไม่ใช้ภาพ · ไม่สร้างข้อความ · 1 forward pass ต่อก้าว",
           "state": "STATE · สิ่งที่โมเดลอ่าน", "choice": "CHOICE · เลือกการกระทำถัดไป", "noul": "NOUL · มีศัตรูอยู่ในเป้าหรือไม่", "pyes": "P(ใช่)",
           "lat": "หน่วง {lat:4.0f} ms  ·  {dps:3.1f} ครั้ง/วินาที  ·  ก้าวที่ {step}", "stats": "พลังชีวิต {hp:.0f}   สังหาร {kills:.0f}   โทเคนที่สร้าง: 0",
           "foot": "huggingface.co/iapp/OpenThai-SystemOne · Apache-2.0 · iApp / OpenThaiGPT"},
}


# ----------------------------------------------------------------------------------------------- model backends
class Backend:
    """Answers one (state, questions) request; returns the parsed `answers` dict of the /v1/systemone contract."""

    def __init__(self, args):
        self.kind = args.backend
        self.permutations = args.permutations
        if self.kind == "iapp":
            key = args.api_key or os.environ.get("IAPP_API_KEY")
            if not key:
                sys.exit("Set IAPP_API_KEY (get a key at https://iapp.co.th) or use --backend local / --backend url")
            self.url, self.headers = IAPP_URL, {"apikey": key, "content-type": "application/json"}
        elif self.kind == "url":
            self.url, self.headers = args.url, {"content-type": "application/json"}
            if args.api_key:
                self.headers["apikey"] = args.api_key
        else:
            from openthai_systemone import SystemOneClient  # pip install openthai-systemone
            self.client = SystemOneClient(args.model)
        if self.kind in ("iapp", "url"):
            import requests
            self.session = requests.Session()

    def ask(self, state, questions):
        if self.kind == "local":
            r = self.client.system_one(state, questions, permutations=self.permutations)
            return {k: json.loads(v.model_dump_json()) for k, v in r.answers.items()}
        body = {"state": state, "questions": questions}
        if self.permutations:
            body["permutations"] = self.permutations
        resp = self.session.post(self.url, headers=self.headers, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=30)
        if resp.status_code == 401:
            sys.exit("401 from the API: check IAPP_API_KEY")
        resp.raise_for_status()
        return resp.json()["answers"]


# ----------------------------------------------------------------------------------------------- game state -> text
def rel_bearing(px, py, pa, ox, oy):
    ang = math.degrees(math.atan2(oy - py, ox - px)) - pa
    return (ang + 180) % 360 - 180


def describe_state(game, state, history, stuck, vzd):
    hp = game.get_game_variable(vzd.GameVariable.HEALTH)
    ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    player = next((o for o in state.objects if o.name == "DoomPlayer"), None)
    visible = {lab.object_id for lab in (state.labels or [])}
    enemies, items = [], []
    if player is not None:
        for o in state.objects:
            if o.name == "DoomPlayer":
                continue
            d = math.hypot(o.position_x - player.position_x, o.position_y - player.position_y) / 32.0
            b = rel_bearing(player.position_x, player.position_y, player.angle, o.position_x, o.position_y)
            side = "ahead" if abs(b) < 15 else ("left" if b > 0 else "right")
            if o.name in ENEMY_NAMES:
                enemies.append((d, b, o.id in visible, f"{o.name} {d:.0f}m {side} {b:+.0f}deg{' VISIBLE' if o.id in visible else ''}"))
            elif "Armor" in o.name or "Medikit" in o.name or "Stimpack" in o.name:
                items.append((d, f"{o.name} {d:.0f}m {side} {b:+.0f}deg"))
    enemies.sort(); items.sort()
    aim = "crosshair: empty"
    if enemies:
        d0, b0, vis0, e0 = enemies[0]
        if vis0 and abs(b0) <= 8:
            aim = f"crosshair: ON {e0.split()[0]} ({b0:+.0f}deg) -> ATTACK would hit"
        else:
            aim = f"crosshair: empty; nearest {'visible ' if vis0 else ''}enemy {abs(b0):.0f}deg to the {'LEFT' if b0 > 0 else 'RIGHT'}"
    wall = ""
    if state.depth_buffer is not None:
        h, w = state.depth_buffer.shape
        c = float(state.depth_buffer[h // 2 - 5 : h // 2 + 5, w // 2 - 10 : w // 2 + 10].mean())
        wall = f"depth ahead: {c:.0f}/255 ({'wall very close' if c < 12 else 'open' if c > 40 else 'obstacle near'})"
    lines = [f"health {hp:.0f}/100 | ammo {ammo:.0f} | kills {kills:.0f}", aim,
             "enemies: " + ("; ".join(e for *_, e in enemies[:4]) if enemies else "none in range"),
             "items: " + ("; ".join(i for _, i in items[:2]) if items else "none"), wall,
             "last actions: " + (" ".join(history) if history else "none") + (" | STUCK: position unchanged" if stuck else "")]
    return "\n".join(l for l in lines if l)


# ----------------------------------------------------------------------------------------------- HUD rendering
def load_fonts():
    reg, bold = HERE / "fonts" / "Sarabun-Regular.ttf", HERE / "fonts" / "Sarabun-Bold.ttf"
    try:
        return ImageFont.truetype(str(bold), 34), ImageFont.truetype(str(bold), 22), ImageFont.truetype(str(reg), 19)
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


def compose(screen, fonts, hud, text, probs, chosen, threat, latency_ms, dps, step, hp, kills, scenario, skill):
    W, H, PW = 1280, 1080, 640
    f_h, f_m, f_s = fonts
    canvas = Image.new("RGB", (W + PW, H), (0, 0, 0))
    canvas.paste(Image.fromarray(screen).resize((1280, 960), Image.NEAREST), (0, 60))
    d = ImageDraw.Draw(canvas)
    cx, cy = W // 2, 60 + 480
    d.line([(cx - 18, cy), (cx + 18, cy)], fill=(255, 255, 255), width=3); d.line([(cx, cy - 18), (cx, cy + 18)], fill=(255, 255, 255), width=3)
    d.rectangle([0, 0, W, 60], fill=(10, 10, 14)); d.rectangle([0, 1020, W, 1080], fill=(10, 10, 14))
    d.text((24, 14), f"> {chosen}", font=f_m, fill=(86, 214, 120)); d.text((24, 1034), f"ViZDoom · {scenario} · skill {skill}", font=f_s, fill=(150, 150, 165))
    x0, pad = W, 26
    d.rectangle([x0, 0, x0 + PW, H], fill=(18, 18, 24))
    d.text((x0 + pad, 26), hud["title"], font=f_h, fill=(255, 255, 255)); d.text((x0 + pad, 74), hud["sub"], font=f_s, fill=(150, 150, 165))
    y = 130
    d.text((x0 + pad, y), hud["state"], font=f_m, fill=(120, 190, 255)); y += 34
    for line in text.split("\n"):
        for chunk in [line[i : i + 64] for i in range(0, len(line), 64)] or [""]:
            d.text((x0 + pad, y), chunk, font=f_s, fill=(225, 225, 225)); y += 26
    y += 24
    d.text((x0 + pad, y), hud["choice"], font=f_m, fill=(120, 190, 255)); y += 36
    bx, bw = x0 + 220, PW - 320
    for name, p in probs:
        col = (86, 214, 120) if name == chosen else (70, 70, 90)
        d.text((x0 + pad, y + 2), name, font=f_s, fill=(255, 255, 255) if name == chosen else (170, 170, 185))
        d.rectangle([bx, y, bx + bw, y + 22], fill=(38, 38, 48)); d.rectangle([bx, y, bx + int(bw * p), y + 22], fill=col)
        d.text((bx + bw + 10, y + 1), f"{100*p:3.0f}%", font=f_s, fill=(220, 220, 220)); y += 32
    y += 18
    d.text((x0 + pad, y), hud["noul"], font=f_m, fill=(120, 190, 255)); y += 36
    d.rectangle([bx, y, bx + bw, y + 22], fill=(38, 38, 48)); d.rectangle([bx, y, bx + int(bw * threat), y + 22], fill=(230, 80, 80) if threat > 0.5 else (90, 90, 110))
    d.text((x0 + pad, y + 2), hud["pyes"], font=f_s, fill=(200, 200, 210)); d.text((bx + bw + 10, y + 1), f"{100*threat:3.0f}%", font=f_s, fill=(220, 220, 220)); y += 56
    d.text((x0 + pad, y), hud["lat"].format(lat=latency_ms, dps=dps, step=step), font=f_m, fill=(255, 210, 90)); y += 38
    d.text((x0 + pad, y), hud["stats"].format(hp=hp, kills=kills), font=f_m, fill=(200, 200, 210))
    d.text((x0 + pad, H - 44), hud["foot"], font=f_s, fill=(120, 120, 140))
    return canvas


# ----------------------------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="iapp", choices=["iapp", "url", "local"])
    ap.add_argument("--api-key", default=None, help="iApp API key (or env IAPP_API_KEY)")
    ap.add_argument("--url", default="http://localhost:8000/v1/systemone", help="for --backend url")
    ap.add_argument("--model", default="iapp/OpenThai-SystemOne", help="for --backend local")
    ap.add_argument("--scenario", default="deadly_corridor", choices=sorted(SCENARIOS))
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--tics", type=int, default=4, help="game tics per decision (35 tics = 1 s); raise to 6-8 on a slow connection")
    ap.add_argument("--lang", default="en", choices=["en", "th"])
    ap.add_argument("--seconds", type=float, default=0, help="stop after this much game time (0 = play until Ctrl-C)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=0, help="order-invariant mode (e.g. 4); 0 = server default")
    ap.add_argument("--no-window", action="store_true", help="do not open the live HUD window")
    ap.add_argument("--record", default="", help="also write an mp4 (needs ffmpeg on PATH)")
    args = ap.parse_args()

    import vizdoom as vzd
    backend = Backend(args)
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{args.scenario}.cfg"))
    game.set_window_visible(False)  # we show our own composite window
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_objects_info_enabled(True); game.set_labels_buffer_enabled(True); game.set_depth_buffer_enabled(True)
    game.set_doom_skill(args.skill); game.set_seed(args.seed)
    game.init()
    buttons = game.get_available_buttons()
    button_of = {b.name: i for i, b in enumerate(buttons)}
    criteria = {n: desc for n, desc in ACTIONS.items() if n in button_of}
    goal, situation = SCENARIOS[args.scenario]
    instructions = f"{situation} {POLICY.format(goal=goal)} Pick the single best next action."
    questions = {"action": {"type": "choice", "instructions": instructions, "criteria": criteria}, "threat": {"type": "noul", "instructions": THREAT_Q}}

    fonts, hud = load_fonts(), HUD[args.lang]
    cv2 = None
    if not args.no_window:
        try:
            import cv2  # noqa: F401
            cv2.namedWindow("OpenThai-SystemOne · Doom", cv2.WINDOW_NORMAL); cv2.resizeWindow("OpenThai-SystemOne · Doom", 1280, 720)
        except Exception:
            print("opencv-python not installed: no live window (pip install opencv-python). Recording/terminal output still works.")
            cv2 = None
    ff = None
    if args.record:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "35", "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", args.record], stdin=subprocess.PIPE)
    history, lat_hist = deque(maxlen=4), deque(maxlen=20)
    frames = step = 0; max_frames = int(args.seconds * 35) if args.seconds else None
    print(f"backend={args.backend} scenario={args.scenario} skill={args.skill} tics/decision={args.tics}  (Ctrl-C to stop)")
    try:
        while max_frames is None or frames < max_frames:
            game.new_episode(); last_pos = None
            while not game.is_episode_finished() and (max_frames is None or frames < max_frames):
                st = game.get_state()
                player = next((o for o in st.objects if o.name == "DoomPlayer"), None)
                pos = (round(player.position_x), round(player.position_y)) if player else None
                stuck = pos is not None and pos == last_pos and history and history[-1].startswith("MOVE"); last_pos = pos
                text = describe_state(game, st, list(history), stuck, vzd)
                t0 = time.perf_counter()
                ans = backend.ask({"game": "Doom", "scenario": args.scenario, "state": text}, questions)
                latency = (time.perf_counter() - t0) * 1000; lat_hist.append(latency)
                chosen = ans["action"]["choice"]; threat = float(ans["threat"]["noul"])
                probs = sorted(ans["action"]["probabilities"].items(), key=lambda kv: -kv[1])
                history.append(chosen); step += 1; dps = 1000 / (sum(lat_hist) / len(lat_hist))
                vec = [0] * len(buttons); vec[button_of[chosen]] = 1; game.set_action(vec)
                hp, kills = game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                if cv2 is None and ff is None:
                    print(f"step {step:4d} {chosen:13s} {100*probs[0][1]:3.0f}%  threat {100*threat:3.0f}%  {latency:4.0f} ms  hp {hp:.0f} kills {kills:.0f}")
                for _ in range(args.tics):
                    if game.is_episode_finished():
                        break
                    game.advance_action(1)
                    s2 = game.get_state(); screen = s2.screen_buffer if s2 is not None else st.screen_buffer
                    if cv2 is not None or ff is not None:
                        frame = compose(screen, fonts, hud, text, probs, chosen, threat, latency, dps, step, hp, kills, args.scenario, args.skill)
                        if ff is not None:
                            ff.stdin.write(frame.tobytes())
                        if cv2 is not None:
                            cv2.imshow("OpenThai-SystemOne · Doom", cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                raise KeyboardInterrupt
                    frames += 1
            print(f"episode over: kills={game.get_game_variable(vzd.GameVariable.KILLCOUNT):.0f} health={game.get_game_variable(vzd.GameVariable.HEALTH):.0f} steps={step}")
    except KeyboardInterrupt:
        pass
    finally:
        if ff is not None:
            ff.stdin.close(); ff.wait(); print("wrote", args.record)
        if cv2 is not None:
            cv2.destroyAllWindows()
        game.close()
        if lat_hist:
            print(f"{step} decisions, mean latency {sum(lat_hist)/len(lat_hist):.0f} ms")


if __name__ == "__main__":
    main()
