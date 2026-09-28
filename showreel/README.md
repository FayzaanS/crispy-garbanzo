# Showreel

A 15-second motion-graphics showreel (1920×1080, 60 fps) in the portfolio's brand: purple / ink / amber,
Alumni Sans + Gasoek One + Permanent Marker, thick outlines and pop shadows.

**Watch:** [`showreel.mp4`](showreel.mp4) · poster frame: [`poster.png`](poster.png)

| Time | Chapter | What's on screen |
| --- | --- | --- |
| 0–2 s | Intro | Beat-synced hits: Consulting → Operations → Biz Dev → +AI, diving through the "I" |
| 2–4.5 s | Hello | Name drop, role stickers, BComm · Ted Rogers School of Management · TMU |
| 4.5–7.5 s | Experience | Timeline: Global Bridging UAE → Maison Edge Realty, intern → associate |
| 7.5–10 s | Toolkit | Live forecast model + twelve skill chips piling up |
| 10–12.5 s | AI Builder | "I build with AI, not just talk about it", 12 shipped projects scrolling past |
| 12.5–15 s | Connect | Name lockup, LinkedIn + GitHub, "Vitam impendere vero" stamp |

## Files

- `index.html` — the whole animation. Every frame is a pure function of time (`window.__render(t)`).
  Open it in a browser for a live, looping preview (click to play with sound).
- `audio.py` — synthesizes the 120 bpm soundtrack and the sound effects, timed from the page's own cue list.
- `render.mjs` — steps through the timeline in headless Chromium, blends 5 sub-frames per frame for motion blur,
  adds grain, and muxes the audio into `showreel.mp4`.
- `fonts/` — the brand fonts (SIL Open Font License), loaded locally so renders are deterministic.

## Re-render

Requires Node with Playwright, Python 3 with `numpy` + `scipy`, and `ffmpeg` (or set `FFMPEG=/path/to/ffmpeg`).

```sh
node render.mjs                                  # 60 fps, 5 sub-frames, 3 workers → showreel.mp4
node render.mjs --fps 30 --samples 1 --out build/draft.mp4   # quick draft
```
