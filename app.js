// Lens Search — ブラウザ内で CLIP 画像埋め込みを計算し、登録済みライブラリから類似画像を検索する
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.0';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const THUMB_MAX = 512;
const TOP_K = 30;

const $ = (id) => document.getElementById(id);

// ---------- モデル ----------
let modelPromise = null;
function loadModel() {
  if (modelPromise) return modelPromise;
  const status = $('modelStatus');
  const files = {};
  const progress_callback = (p) => {
    if (p.status === 'progress' && p.total) {
      files[p.file] = [p.loaded, p.total];
      const [l, t] = Object.values(files).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      status.textContent = `モデル読込中 ${Math.round((l / t) * 100)}%`;
    }
  };
  status.textContent = 'モデル読込中…';
  modelPromise = Promise.all([
    AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8', progress_callback }),
  ]).then(([processor, model]) => {
    status.textContent = 'モデル準備完了';
    status.classList.add('ready');
    return { processor, model };
  }).catch((e) => {
    modelPromise = null;
    status.textContent = 'モデル読込失敗';
    throw e;
  });
  return modelPromise;
}

async function embed(blob) {
  const { processor, model } = await loadModel();
  const image = await RawImage.fromBlob(blob);
  const inputs = await processor(image);
  const { image_embeds } = await model(inputs);
  const v = new Float32Array(image_embeds.data);
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

// ---------- 画像ユーティリティ ----------
async function toBitmap(blob) {
  try { return await createImageBitmap(blob, { imageOrientation: 'from-image' }); }
  catch { return await createImageBitmap(blob); }
}

function canvasToBlob(canvas, quality = 0.85) {
  return new Promise((res) => canvas.toBlob(res, 'image/jpeg', quality));
}

// source（ImageBitmap / video / img）の指定領域を最大 THUMB_MAX px に縮小して JPEG 化
async function resizeToBlob(source, sx, sy, sw, sh) {
  const scale = Math.min(1, THUMB_MAX / Math.max(sw, sh));
  const c = document.createElement('canvas');
  c.width = Math.round(sw * scale);
  c.height = Math.round(sh * scale);
  c.getContext('2d').drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return canvasToBlob(c);
}

// ---------- IndexedDB ----------
const dbPromise = new Promise((res, rej) => {
  const req = indexedDB.open('lens-search', 1);
  req.onupgradeneeded = () => req.result.createObjectStore('images', { keyPath: 'id', autoIncrement: true });
  req.onsuccess = () => res(req.result);
  req.onerror = () => rej(req.error);
});

async function tx(mode, fn) {
  const db = await dbPromise;
  return new Promise((res, rej) => {
    const t = db.transaction('images', mode);
    const r = fn(t.objectStore('images'));
    t.oncomplete = () => res(r?.result);
    t.onerror = () => rej(t.error);
  });
}
const dbAll = () => tx('readonly', (s) => s.getAll());
const dbAdd = (rec) => tx('readwrite', (s) => s.add(rec));
const dbDel = (id) => tx('readwrite', (s) => s.delete(id));
const dbClear = () => tx('readwrite', (s) => s.clear());

// メモリ上のライブラリ { id, name, thumb(Blob), url, emb(Float32Array) }
let library = [];

async function loadLibrary() {
  library.forEach((r) => URL.revokeObjectURL(r.url));
  const rows = await dbAll();
  library = rows.map((r) => ({ ...r, emb: new Float32Array(r.emb), url: URL.createObjectURL(r.thumb) }));
  renderLibrary();
}

// ---------- ライブラリ UI ----------
function card(rec, { score, onDelete } = {}) {
  const el = document.createElement('div');
  el.className = 'card';
  const img = document.createElement('img');
  img.src = rec.url;
  img.alt = rec.name || '';
  img.loading = 'lazy';
  img.onclick = () => openViewer(rec.url);
  el.append(img);
  if (score != null) {
    const s = document.createElement('span');
    s.className = 'score';
    s.textContent = `${(score * 100).toFixed(1)}%`;
    el.append(s);
  }
  if (onDelete) {
    const b = document.createElement('button');
    b.className = 'del';
    b.textContent = '×';
    b.title = '削除';
    b.onclick = onDelete;
    el.append(b);
  }
  return el;
}

function renderLibrary() {
  $('libCount').textContent = library.length;
  const grid = $('libGrid');
  grid.replaceChildren();
  if (!library.length) {
    grid.innerHTML = '<p class="empty" style="grid-column:1/-1">まだ画像がありません。「＋ 画像を追加」から写真フォルダの画像をまとめて登録してください。</p>';
    return;
  }
  for (const rec of [...library].reverse()) {
    // サンプルは samples/ フォルダで管理するので画面からは消さない
    grid.append(card(rec, rec.source?.startsWith('samples/') ? {} : {
      onDelete: async () => {
        await dbDel(rec.id);
        URL.revokeObjectURL(rec.url);
        library = library.filter((r) => r.id !== rec.id);
        renderLibrary();
      },
    }));
  }
}

// items: { name, source?, getBlob: () => Promise<Blob> }[]
async function addImages(items, label = '解析中') {
  if (!items.length) return 0;
  const prog = $('progress');
  prog.hidden = false;
  let done = 0, failed = 0;
  const update = () => {
    $('progressBar').style.width = `${(done / items.length) * 100}%`;
    $('progressText').textContent = `${label} ${done} / ${items.length}`;
  };
  update();
  for (const item of items) {
    try {
      const bmp = await toBitmap(await item.getBlob());
      const thumb = await resizeToBlob(bmp, 0, 0, bmp.width, bmp.height);
      bmp.close?.();
      const emb = await embed(thumb);
      const rec = { name: item.name, source: item.source, thumb, emb: emb.buffer, addedAt: Date.now() };
      rec.id = await dbAdd(rec);
      library.push({ ...rec, emb, url: URL.createObjectURL(thumb) });
    } catch (err) {
      console.error(item.name, err);
      failed++;
    }
    done++;
    update();
  }
  prog.hidden = true;
  renderLibrary();
  return failed;
}

$('libInput').addEventListener('change', async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  const failed = await addImages(files.map((f) => ({ name: f.name, getBlob: async () => f })));
  if (failed) alert(`${failed} 枚の画像を読み込めませんでした`);
});

// samples/ フォルダの画像をライブラリに同期（追加・差し替え・削除）
const IMAGE_RE = /\.(jpe?g|png|webp|gif|avif|bmp)$/i;

// 画像一覧 [{ name, size }] を取得。GitHub Pages では GitHub API、ローカルでは serve.rb の samples.json
async function listSamples() {
  const host = location.hostname;
  if (host.endsWith('.github.io')) {
    const owner = host.split('.')[0];
    const repo = location.pathname.split('/')[1] || `${owner}.github.io`;
    const res = await fetch(`https://api.github.com/repos/${owner}/${repo}/contents/samples`);
    if (res.status === 404) return [];
    if (!res.ok) throw new Error(`GitHub API ${res.status}`);
    return (await res.json())
      .filter((f) => f.type === 'file' && IMAGE_RE.test(f.name))
      .map((f) => ({ name: f.name, size: f.size }));
  }
  const res = await fetch('samples.json', { cache: 'no-store' });
  if (!res.ok) throw new Error(`samples.json ${res.status}`);
  return res.json();
}

async function syncSamples() {
  let list;
  try {
    list = await listSamples();
  } catch (err) {
    console.warn('サンプル一覧を取得できませんでした', err);
    return;
  }
  const key = (f) => `samples/${f.name}#${f.size}`;
  const wanted = new Set(list.map(key));
  for (const rec of library.filter((r) => r.source?.startsWith('samples/') && !wanted.has(r.source))) {
    await dbDel(rec.id);
    URL.revokeObjectURL(rec.url);
  }
  library = library.filter((r) => !r.source?.startsWith('samples/') || wanted.has(r.source));
  const have = new Set(library.map((r) => r.source));
  const todo = list.filter((f) => !have.has(key(f))).map((f) => ({
    name: f.name,
    source: key(f),
    getBlob: async () => {
      const r = await fetch(`samples/${encodeURIComponent(f.name)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.blob();
    },
  }));
  await addImages(todo, 'サンプル登録中');
  renderLibrary();
}

$('clearLib').addEventListener('click', async () => {
  if (!library.length) return;
  const btn = $('clearLib');
  if (btn.dataset.armed !== '1') {
    btn.dataset.armed = '1';
    btn.textContent = '本当に削除？';
    setTimeout(() => { btn.dataset.armed = ''; btn.textContent = '全削除'; }, 3000);
    return;
  }
  btn.dataset.armed = '';
  btn.textContent = '全削除';
  await dbClear();
  await loadLibrary();
});

// ---------- カメラ ----------
const video = $('video');
let stream = null;
let capturedBitmap = null; // 現在の検索対象画像

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    $('noCamera').hidden = false;
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1440 } },
      audio: false,
    });
    video.srcObject = stream;
    video.hidden = false;
    $('shutter').hidden = false;
    $('noCamera').hidden = true;
  } catch (err) {
    console.warn('camera unavailable', err);
    $('noCamera').hidden = false;
    $('noCamera').querySelector('p').textContent = 'カメラを起動できませんでした。下のボタンから撮影・選択してください。';
  }
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.hidden = true;
  $('shutter').hidden = true;
}

$('shutter').addEventListener('click', async () => {
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  showCaptured(await toBitmap(await canvasToBlob(c, 0.92)));
});

$('fileInput').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (f) showCaptured(await toBitmap(f));
});

$('retake').addEventListener('click', () => {
  capturedBitmap?.close?.();
  capturedBitmap = null;
  $('cropWrap').hidden = true;
  $('retake').hidden = true;
  $('searchBtn').hidden = true;
  $('results').replaceChildren();
  $('hint').textContent = '';
  startCamera();
});

async function showCaptured(bmp) {
  stopCamera();
  $('noCamera').hidden = true;
  capturedBitmap = bmp;
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  const img = $('captured');
  img.src = URL.createObjectURL(await canvasToBlob(c, 0.9));
  img.onload = () => URL.revokeObjectURL(img.src);
  $('cropWrap').hidden = false;
  $('cropBox').hidden = true;
  crop = null;
  $('retake').hidden = false;
  $('searchBtn').hidden = false;
  $('hint').textContent = '画像をドラッグして範囲を選ぶと、その部分だけで検索します';
  runSearch();
}

// ---------- 範囲選択 ----------
let crop = null; // 表示座標系での選択範囲 {x, y, w, h}
let dragStart = null;
const wrap = $('cropWrap');

wrap.addEventListener('pointerdown', (e) => {
  const r = wrap.getBoundingClientRect();
  dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
  wrap.setPointerCapture(e.pointerId);
});
wrap.addEventListener('pointermove', (e) => {
  if (!dragStart) return;
  const r = wrap.getBoundingClientRect();
  const x = Math.max(0, Math.min(r.width, e.clientX - r.left));
  const y = Math.max(0, Math.min(r.height, e.clientY - r.top));
  crop = {
    x: Math.min(x, dragStart.x), y: Math.min(y, dragStart.y),
    w: Math.abs(x - dragStart.x), h: Math.abs(y - dragStart.y),
  };
  const box = $('cropBox');
  box.hidden = false;
  Object.assign(box.style, { left: `${crop.x}px`, top: `${crop.y}px`, width: `${crop.w}px`, height: `${crop.h}px` });
});
wrap.addEventListener('pointerup', () => {
  dragStart = null;
  if (!crop || crop.w < 16 || crop.h < 16) {
    crop = null;
    $('cropBox').hidden = true;
  } else {
    runSearch();
  }
});

$('searchBtn').addEventListener('click', runSearch);

// ---------- 検索 ----------
let searchSeq = 0;
async function runSearch() {
  if (!capturedBitmap) return;
  const seq = ++searchSeq;
  const results = $('results');
  if (!library.length) {
    results.innerHTML = '<p class="empty">ライブラリが空です。「ライブラリ」タブで検索対象の画像を登録してください。</p>';
    return;
  }
  results.innerHTML = '<p class="empty">解析中…</p>';
  $('searchBtn').disabled = true;
  try {
    const bmp = capturedBitmap;
    let sx = 0, sy = 0, sw = bmp.width, sh = bmp.height;
    if (crop) {
      const r = wrap.getBoundingClientRect();
      const kx = bmp.width / r.width, ky = bmp.height / r.height;
      sx = crop.x * kx; sy = crop.y * ky; sw = crop.w * kx; sh = crop.h * ky;
    }
    const q = await embed(await resizeToBlob(bmp, sx, sy, sw, sh));
    if (seq !== searchSeq) return;

    const scored = library.map((rec) => {
      let s = 0;
      for (let i = 0; i < q.length; i++) s += q[i] * rec.emb[i];
      return { rec, s };
    }).sort((a, b) => b.s - a.s).slice(0, TOP_K);

    const h = document.createElement('h3');
    h.textContent = `類似画像 ${scored.length} 件${crop ? '（選択範囲）' : ''}`;
    const grid = document.createElement('div');
    grid.className = 'grid';
    scored.forEach(({ rec, s }) => grid.append(card(rec, { score: s })));
    results.replaceChildren(h, grid);
  } catch (err) {
    console.error(err);
    results.innerHTML = `<p class="empty">検索に失敗しました: ${err.message}</p>`;
  } finally {
    $('searchBtn').disabled = false;
  }
}

// ---------- ビューア / タブ ----------
function openViewer(url) {
  const v = document.createElement('div');
  v.className = 'viewer';
  v.innerHTML = `<img src="${url}" alt="">`;
  v.onclick = () => v.remove();
  document.body.append(v);
}

document.querySelectorAll('.tabbar button').forEach((b) => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.tabbar button').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${b.dataset.view}`));
    if (b.dataset.view === 'search' && !capturedBitmap) startCamera();
    else if (b.dataset.view !== 'search') stopCamera();
  });
});

// ---------- 起動 ----------
loadLibrary().then(syncSamples);
startCamera();
loadModel().catch((e) => console.error(e));
