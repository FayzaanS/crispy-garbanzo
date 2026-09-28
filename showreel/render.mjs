// Render index.html to showreel.mp4 — frame-accurate, with motion blur and soundtrack.
//
//   node render.mjs [--fps 60] [--samples 5] [--workers 3] [--out showreel.mp4]
//
// Needs Playwright (resolved locally or from `npm root -g`), python3 with
// numpy + scipy (for audio.py), and ffmpeg on PATH (or set FFMPEG=/path/to/ffmpeg).
//
// Each output frame is the average of `samples` sub-frames spread across a
// 180° shutter, which gives real motion blur on the fast moves. Workers each
// render a contiguous chunk to a lossless intermediate; the chunks are then
// joined, film grain is added, and the soundtrack is muxed in.

import { spawn, execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUILD = path.join(HERE, 'build');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith('--') ? [...a, [v.slice(2), all[i + 1]?.startsWith('--') || all[i + 1] === undefined ? true : all[i + 1]]] : a), []));
const FPS = +(args.fps ?? 60);
const SAMPLES = +(args.samples ?? 5);
const WORKERS = +(args.workers ?? 3);
const SHUTTER = 0.5; // fraction of a frame the virtual shutter is open
const OUT = path.resolve(HERE, args.out ?? 'showreel.mp4');

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch { /* fall through to global install */ }
  return require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
}
const { chromium } = loadPlaywright();
const PAGE_URL = 'file://' + path.join(HERE, 'index.html') + '?render=1';

function run(cmd, argv, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, argv, { stdio: ['pipe', 'inherit', 'inherit'], ...opts });
    p.on('error', reject);
    p.on('close', code => (code === 0 ? resolve(p) : reject(new Error(`${cmd} exited with ${code}`))));
    if (opts.onSpawn) opts.onSpawn(p);
  });
}

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => { throw e; });
  await page.goto(PAGE_URL);
  await page.evaluate(() => window.__ready);
  await page.addStyleTag({ content: '#grain{display:none!important}' }); // grain is added by ffmpeg instead
  return page;
}

async function renderChunk(id, f0, f1, duration) {
  const browser = await chromium.launch();
  const page = await openPage(browser);
  const cdp = await page.context().newCDPSession(page);
  const file = path.join(BUILD, `chunk${id}.mkv`);
  const ff = spawn(FFMPEG, [
    '-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(FPS * SAMPLES), '-c:v', 'png', '-i', '-',
    '-vf', `tmix=frames=${SAMPLES},select='eq(mod(n\\,${SAMPLES})\\,${SAMPLES - 1})',setpts=N/(${FPS}*TB),scale=out_color_matrix=bt709:out_range=tv,format=yuv444p`,
    '-r', String(FPS), '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast', file,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => { ff.on('close', c => (c === 0 ? res() : rej(new Error(`ffmpeg chunk ${id} exited ${c}`)))); ff.on('error', rej); });
  const t0 = Date.now();
  for (let f = f0; f < f1; f++) {
    for (let j = 0; j < SAMPLES; j++) {
      const off = SAMPLES === 1 ? 0 : ((j + 0.5) / SAMPLES - 0.5) * SHUTTER / FPS;
      const t = Math.min(duration - 1e-4, Math.max(0, f / FPS + off));
      await page.evaluate(t => window.__render(t), t);
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
      if (!ff.stdin.write(Buffer.from(shot.data, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    }
    if ((f - f0) % 60 === 59) console.log(`  worker ${id}: ${f - f0 + 1}/${f1 - f0} frames (${((Date.now() - t0) / (f - f0 + 1)).toFixed(0)} ms/frame)`);
  }
  ff.stdin.end();
  await done;
  await browser.close();
  return file;
}

async function main() {
  fs.mkdirSync(BUILD, { recursive: true });

  // 1. sound cues + poster from the page itself
  const browser = await chromium.launch();
  const page = await openPage(browser);
  const cues = await page.evaluate(() => window.__sfx());
  const duration = await page.evaluate(() => window.__duration);
  fs.writeFileSync(path.join(BUILD, 'cues.json'), JSON.stringify(cues, null, 1));
  await page.addStyleTag({ content: '#grain{display:block!important}' });
  await page.evaluate(t => window.__render(t), duration - 0.01);
  await page.screenshot({ path: path.join(HERE, 'poster.png') });
  await browser.close();
  console.log(`${cues.length} sound cues, ${duration}s @ ${FPS}fps, ${SAMPLES} sub-frames, ${WORKERS} workers`);

  // 2. soundtrack
  const wav = path.join(BUILD, 'audio.wav');
  await run('python3', [path.join(HERE, 'audio.py'), path.join(BUILD, 'cues.json'), wav]);
  fs.copyFileSync(wav, path.join(HERE, 'audio.wav')); // lets the live preview play with sound

  // 3. frames
  const total = Math.round(duration * FPS);
  const per = Math.ceil(total / WORKERS);
  const t0 = Date.now();
  const chunks = await Promise.all(Array.from({ length: WORKERS }, (_, i) => renderChunk(i, i * per, Math.min(total, (i + 1) * per), duration)));
  console.log(`frames done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  // 4. join, grain, mux
  const list = path.join(BUILD, 'chunks.txt');
  fs.writeFileSync(list, chunks.map(c => `file '${c}'`).join('\n'));
  await run(FFMPEG, [
    '-y', '-loglevel', 'error',
    '-f', 'concat', '-safe', '0', '-i', list, '-i', wav,
    '-vf', 'noise=c0s=7:c0f=u,format=yuv420p',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-profile:v', 'high', '-r', String(FPS),
    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
    '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', '-shortest', OUT,
  ]);
  console.log(`wrote ${OUT}`);
}

main().catch(e => { console.error(e); process.exit(1); });
