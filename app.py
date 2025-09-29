#!/usr/bin/env python3
"""
Flask Stego Web App

What it does
------------
- Lets you upload a base PNG + message + password and creates a stego PNG on the server.
- Gives you a shareable link like /v/<id>. When someone visits it, they see a password prompt, enter the password, and the server returns the decrypted message.
- Also supports uploading an already-embedded stego PNG (made by the CLI script I gave you). You then just share the link; recipients will only need the password to view the message.

How to run
----------
1) pip install flask pillow cryptography
2) python app.py
3) Open http://127.0.0.1:5000

Notes
-----
- Uses the same header format as the autodetect CLI (MAGIC b"STG1").
- Uses Fernet (PBKDF2->AES128-CBC+HMAC) compatible with the Python scripts provided earlier.
- Stores images on disk under ./data and never stores plaintext messages.

"""
import os
import secrets
import base64
import gzip
import struct
from io import BytesIO
from typing import Iterable, List, Tuple

from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template_string, abort
from PIL import Image
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

# ----------------- Config -----------------
APP_TITLE = "StegoShare"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------- Crypto helpers -----------------
SALT_LEN = 16
PBKDF2_ITERS = 200_000
MAGIC = b"STG1"
HEADER_LEN_BYTES = 9
HEADER_BITS = HEADER_LEN_BYTES * 8


def derive_fernet_key(password: str, salt: bytes, iterations: int = PBKDF2_ITERS) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=default_backend()
    )
    key = kdf.derive(password.encode("utf-8"))
    return base64.urlsafe_b64encode(key)


def encrypt_payload(password: str, plaintext_bytes: bytes) -> bytes:
    salt = secrets.token_bytes(SALT_LEN)
    f = Fernet(derive_fernet_key(password, salt))
    token = f.encrypt(plaintext_bytes)
    return salt + token


def decrypt_payload(password: str, payload: bytes) -> bytes:
    if len(payload) < SALT_LEN:
        raise ValueError("Payload too short (no salt).")
    salt, token = payload[:SALT_LEN], payload[SALT_LEN:]
    f = Fernet(derive_fernet_key(password, salt))
    return f.decrypt(token)


# -------------- (De)compression --------------

def maybe_compress(data: bytes, use: bool) -> bytes:
    if not use:
        return data
    out = BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(data)
    return out.getvalue()


def maybe_decompress(data: bytes, use: bool) -> bytes:
    if not use:
        return data
    with gzip.GzipFile(fileobj=BytesIO(data), mode="rb") as gz:
        return gz.read()


# ----------------- Bit helpers -----------------

def bytes_to_bits(data: bytes) -> Iterable[int]:
    for b in data:
        for i in range(7, -1, -1):
            yield (b >> i) & 1


def bits_to_bytes(bits_iter) -> bytes:
    out = bytearray()
    b = 0
    cnt = 0
    for bit in bits_iter:
        b = (b << 1) | (bit & 1)
        cnt += 1
        if cnt == 8:
            out.append(b)
            b = 0
            cnt = 0
    return bytes(out)


# ----------------- Header helpers -----------------
# FLAGS layout:
# bits 0-1: (lsb-1)
# bit 2: use_alpha
# bit 3: compressed
# bits 4-7: reserved (0)

def build_header(payload_len: int, lsb: int, use_alpha: bool, compressed: bool) -> bytes:
    if not (1 <= lsb <= 3):
        raise ValueError("lsb must be 1..3")
    flags = ((lsb - 1) & 0b11) | ((1 if use_alpha else 0) << 2) | ((1 if compressed else 0) << 3)
    return MAGIC + struct.pack(">B I", flags, payload_len)


def parse_header(header: bytes) -> Tuple[int, bool, bool, int]:
    if len(header) != HEADER_LEN_BYTES or header[:4] != MAGIC:
        raise ValueError("Invalid or missing header (wrong file or corrupted).")
    flags, data_len = struct.unpack(">B I", header[4:9])
    lsb = (flags & 0b11) + 1
    use_alpha = bool((flags >> 2) & 1)
    compressed = bool((flags >> 3) & 1)
    return lsb, use_alpha, compressed, data_len


# ----------------- Channel helpers -----------------

def to_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def flatten_rgba(img: Image.Image) -> List[int]:
    return [c for px in img.getdata() for c in (px if len(px) == 4 else (*px, 255))]


def unflatten_rgba(flat: List[int], size: Tuple[int, int]) -> Image.Image:
    it = iter(flat)
    pixels = [tuple(next(it) for _ in range(4)) for _ in range((size[0] * size[1]))]
    out = Image.new("RGBA", size)
    out.putdata(pixels)
    return out


def rgb_channel_indices(num_pixels: int) -> List[int]:
    idx = []
    for i in range(num_pixels):
        base = 4 * i
        idx.extend([base + 0, base + 1, base + 2])
    return idx


def rgba_channel_indices(num_pixels: int) -> List[int]:
    return list(range(num_pixels * 4))


# ----------------- Capacity -----------------

def payload_capacity_bytes(img_size: Tuple[int, int], use_alpha: bool, lsb: int) -> int:
    w, h = img_size
    npx = w * h
    channels_allowed = 4 if use_alpha else 3
    total_channels = npx * channels_allowed
    usable_channels = total_channels - 72  # header uses first 72 RGB slots
    if usable_channels < 0:
        return 0
    return (usable_channels * lsb) // 8


# ----------------- Stego core -----------------

def embed_into_png(in_path: str, out_path: str, encrypted: bytes, lsb: int, use_alpha: bool, compressed: bool):
    img = to_rgba(Image.open(in_path))
    w, h = img.size
    npx = w * h
    flat = flatten_rgba(img)

    # Header into first 72 RGB LSB slots
    rgb_idx = rgb_channel_indices(npx)
    if len(rgb_idx) < HEADER_BITS:
        raise ValueError("Image too small to store header.")
    header = build_header(len(encrypted), lsb, use_alpha, compressed)
    header_bits = list(bytes_to_bits(header))
    header_slots = rgb_idx[:HEADER_BITS]
    for slot, bit in zip(header_slots, header_bits):
        flat[slot] = (flat[slot] & ~1) | bit

    # Payload after last header slot
    allowed_idx = rgb_idx if not use_alpha else rgba_channel_indices(npx)
    last_header_channel = header_slots[-1]
    try:
        start_pos = next(i for i, ch in enumerate(allowed_idx) if ch > last_header_channel)
    except StopIteration:
        raise ValueError("No room for payload after header.")

    payload_bits_iter = bytes_to_bits(encrypted)
    for ch_idx in allowed_idx[start_pos:]:
        new_low = 0
        wrote = 0
        for _ in range(lsb):
            try:
                b = next(payload_bits_iter)
            except StopIteration:
                for __ in range(lsb - wrote):
                    new_low <<= 1
                val = flat[ch_idx]
                flat[ch_idx] = ((val >> lsb) << lsb) | new_low
                out = unflatten_rgba(flat, img.size)
                out.save(out_path, format="PNG")
                return
            new_low = (new_low << 1) | (b & 1)
            wrote += 1
        val = flat[ch_idx]
        flat[ch_idx] = ((val >> lsb) << lsb) | new_low

    out = unflatten_rgba(flat, img.size)
    out.save(out_path, format="PNG")


def extract_from_png(in_path: str) -> Tuple[bytes, int, bool, bool]:
    img = to_rgba(Image.open(in_path))
    w, h = img.size
    npx = w * h
    flat = flatten_rgba(img)

    rgb_idx = rgb_channel_indices(npx)
    if len(rgb_idx) < HEADER_BITS:
        raise ValueError("Image too small or no header present.")
    header_slots = rgb_idx[:HEADER_BITS]
    header_bits = [(flat[idx] & 1) for idx in header_slots]
    header = bits_to_bytes(iter(header_bits))
    lsb, use_alpha, compressed, data_len = parse_header(header)

    allowed_idx = rgb_idx if not use_alpha else rgba_channel_indices(npx)
    last_header_channel = header_slots[-1]
    try:
        start_pos = next(i for i, ch in enumerate(allowed_idx) if ch > last_header_channel)
    except StopIteration:
        raise ValueError("No payload area found after header.")

    bits_needed = data_len * 8
    got_bits = 0
    payload_bits = []

    for ch_idx in allowed_idx[start_pos:]:
        val = flat[ch_idx]
        for i in range(lsb - 1, -1, -1):
            payload_bits.append((val >> i) & 1)
            got_bits += 1
            if got_bits >= bits_needed:
                break
        if got_bits >= bits_needed:
            break

    if got_bits < bits_needed:
        raise ValueError("Not enough embedded data for declared payload.")

    encrypted = bits_to_bytes(iter(payload_bits[:bits_needed]))
    return encrypted, lsb, use_alpha, compressed


# ----------------- Flask app -----------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{{ app_title }}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b1020; color: #e6ebff; }
    header { background:#111733; border-bottom:1px solid #1f2852; padding:12px 20px; }
    header h1 { margin:0; font-size:18px; font-weight:700; }
    .wrap { max-width: 900px; margin: 40px auto; padding: 24px; }
    h1 { font-weight: 700; margin: 0 0 20px; font-size: 28px; }
    .card { background: #111733; border: 1px solid #1f2852; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    label { display:block; font-size:13px; opacity:.9; margin: 10px 0 6px; }
    input[type=file], input[type=text], input[type=password], textarea, select { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b376e; background:#0d1330; color:#e6ebff; }
    textarea { min-height: 100px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .actions { margin-top: 16px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .btn { background:#3355ff; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; }
    .btn.secondary { background:#22306b; }
    .btn.danger { background:#a83333; }
    .small { font-size: 12px; opacity:.75; }
    .muted { opacity:.8; }
    .notice { margin-top:8px; padding:10px; background:#0d1433; border:1px solid #243076; border-radius:10px; }
  </style>
</head>
<body>
  <header>
    <h1>🔐 {{ app_title }}</h1>
  </header>
  <div class="wrap">
    <div class="card">
      <p class="muted">Create a shareable link that hides your message inside a PNG. Viewers will be asked for the password to reveal the text.</p>

      <!-- Create/Import form (standalone) -->
      <form action="/create" method="post" enctype="multipart/form-data">
        <label>Base PNG (lossless)</label>
        <input type="file" name="base" accept="image/png" required>

        <label>Message to hide</label>
        <textarea name="message" placeholder="Type your secret here" required></textarea>

        <div class="row">
          <div>
            <label>Password</label>
            <input type="password" name="password" required>
          </div>
          <div>
            <label>LSBs per channel</label>
            <select name="lsb">
              <option value="1" selected>1 (stealthy)</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </div>
        </div>

        <div class="row">
          <div>
            <label><input type="checkbox" name="use_alpha"> Use alpha channel (more capacity)</label>
          </div>
          <div>
            <label><input type="checkbox" name="no_compress"> Disable compression</label>
          </div>
        </div>

        <div class="actions">
          <button class="btn" type="submit">Create link</button>
          <a class="btn secondary" href="/import">I already have a stego PNG</a>
          <a class="btn secondary" href="/mask">Mask a secret image</a>
        </div>
      </form>

      <!-- Separate cleanup form (NOT nested) -->
      <form class="actions" action="/admin/cleanup" method="post" onsubmit="return confirm('Delete ALL stored images? This cannot be undone.');">
        <button class="btn danger" type="submit">Delete all stored images</button>
      </form>

      <div class="notice small">We never store plaintext; only the stego image file. Decryption happens per-request with the password you provide.</div>
    </div>
  </div>
</body>
</html>
"""

IMPORT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Import • {{ app_title }}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b1020; color: #e6ebff; }
    .wrap { max-width: 700px; margin: 40px auto; padding: 24px; }
    .card { background: #111733; border: 1px solid #1f2852; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    label { display:block; font-size:13px; opacity:.9; margin: 10px 0 6px; }
    input[type=file] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b376e; background:#0d1330; color:#e6ebff; }
    .btn { background:#3355ff; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; margin-top: 14px; }
    a { color:#9bb5ff; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>Import an existing stego PNG</h2>
      <p>Upload a PNG created by the CLI/script, then share the link. Viewers will need the password to decrypt.</p>
      <form action="/import" method="post" enctype="multipart/form-data">
        <label>Stego PNG</label>
        <input type="file" name="stego" accept="image/png" required>
        <button class="btn" type="submit">Upload</button>
      </form>
      <p><a href="/">← Back</a></p>
    </div>
  </div>
</body>
</html>
"""

VIEW_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>View • {{ app_title }}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b1020; color: #e6ebff; }
    header { background:#111733; border-bottom:1px solid #1f2852; padding:12px 20px; }
    header a { color:#9bb5ff; text-decoration:none; font-weight:600; margin-right:20px; }
    header a:hover { text-decoration:underline; }
    .wrap { max-width: 1000px; margin: 40px auto; padding: 24px; }
    .card { background: #111733; border: 1px solid #1f2852; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    .imgbox { width: 100%; }
    .imgbox img { width: 100%; height: auto; display: block; border-radius:12px; border:1px solid #243076; }
    .btn { background:#3355ff; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; }
    .out { white-space: pre-wrap; background:#0d1330; border:1px solid #2b376e; padding:12px; border-radius:10px; margin-top:12px; }
    .small { font-size: 12px; opacity:.75; }
    .err { color:#ff8a8a; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    input[type=password] { flex:1; min-width: 200px; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b376e; background:#0d1330; color:#e6ebff; }
    label.checkbox { display:flex; align-items:center; gap:8px; font-size:13px; opacity:.9; margin-top:8px; }
  </style>
</head>
<body>
  <header>
    <a href="/">← Back to Create</a>
  </header>
  <div class="wrap">
    <div class="card">
      <div class="imgbox"><img src="{{ image_url }}" alt="Shared stego image"></div>
      <div class="row" style="margin-top:14px">
        <input id="pwd" type="password" placeholder="Enter password to reveal">
        <button class="btn" onclick="decrypt()">Reveal</button>
      </div>
      <label class="checkbox"><input id="delafter" type="checkbox"> Delete this image from server after reveal</label>
      <div id="error" class="small err"></div>
      <div id="out" class="out" style="display:none"></div>
      <div class="small" style="margin-top:8px">Keep this page open; nothing is stored except the stego PNG file.</div>
    </div>
  </div>
<script>
async function decrypt() {
  const pwd = document.getElementById('pwd').value;
  const del = document.getElementById('delafter').checked;
  document.getElementById('error').textContent = '';
  const res = await fetch(window.location.pathname.replace('/v/','/api/decrypt/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: pwd, delete_after: del })
  });
  if (!res.ok) {
    const t = await res.text();
    document.getElementById('error').textContent = t || 'Failed to decrypt.';
    return;
  }
  const data = await res.json();
  const out = document.getElementById('out');
  out.style.display = 'block';
  out.textContent = data.message || '[binary payload saved on server]';
  if (data.deleted) {
    const note = document.createElement('div');
    note.className = 'small';
    note.textContent = 'This image has been deleted from the server.';
    out.parentNode.insertBefore(note, out.nextSibling);
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, app_title=APP_TITLE)


@app.route("/create", methods=["POST"])
def create():
    base = request.files.get("base")
    message = request.form.get("message", "")
    password = request.form.get("password", "")
    lsb = int(request.form.get("lsb", 1))
    use_alpha = bool(request.form.get("use_alpha"))
    compress = not bool(request.form.get("no_compress"))

    if not base or not message or not password:
        return "Missing fields", 400

    # Save base temporarily
    tmp_id = secrets.token_urlsafe(8)
    base_path = os.path.join(DATA_DIR, f"{tmp_id}_base.png")
    base.save(base_path)

    # Build encrypted payload
    comp = maybe_compress(message.encode("utf-8"), compress)
    encrypted = encrypt_payload(password, comp)

    # Check capacity and embed
    img = Image.open(base_path)
    cap = payload_capacity_bytes(img.size, use_alpha, lsb)
    if len(encrypted) > cap:
        os.remove(base_path)
        return f"Payload too large for this image. Capacity ~{cap} bytes; payload {len(encrypted)} bytes.", 400

    stego_id = secrets.token_urlsafe(10)
    out_path = os.path.join(DATA_DIR, f"{stego_id}.png")
    embed_into_png(base_path, out_path, encrypted, lsb, use_alpha, compress)

    # remove temp base
    try:
        os.remove(base_path)
    except Exception:
        pass

    return redirect(url_for('view', sid=stego_id))


@app.route("/import", methods=["GET", "POST"])
def import_existing():
    if request.method == "GET":
        return render_template_string(IMPORT_HTML, app_title=APP_TITLE)
    stego = request.files.get("stego")
    if not stego:
        return "No file uploaded", 400
    stego_id = secrets.token_urlsafe(10)
    out_path = os.path.join(DATA_DIR, f"{stego_id}.png")
    stego.save(out_path)
    return redirect(url_for('view', sid=stego_id))


@app.route("/v/<sid>")
def view(sid):
    img_path = os.path.join(DATA_DIR, f"{sid}.png")
    if not os.path.exists(img_path):
        abort(404)
    image_url = url_for('image', filename=f"{sid}.png")
    return render_template_string(VIEW_HTML, image_url=image_url, app_title=APP_TITLE)


@app.route("/api/decrypt/<sid>", methods=["POST"])
def api_decrypt(sid):
    img_path = os.path.join(DATA_DIR, f"{sid}.png")
    if not os.path.exists(img_path):
        return "Not found", 404
    js = request.get_json(silent=True) or {}
    password = js.get("password", "")
    delete_after = bool(js.get("delete_after", False))
    if not password:
        return "Password required", 400
    try:
        encrypted, lsb, use_alpha, compressed = extract_from_png(img_path)
        decrypted = decrypt_payload(password, encrypted)
        if compressed:
            try:
                decrypted = maybe_decompress(decrypted, True)
            except Exception:
                pass
        # Try UTF-8
        message = None
        try:
            message = decrypted.decode("utf-8")
        except UnicodeDecodeError:
            # Save binary to disk (rare)
            bin_path = os.path.join(DATA_DIR, f"{sid}_extracted.bin")
            with open(bin_path, "wb") as f:
                f.write(decrypted)
        # Optionally delete the image file after successful reveal
        deleted = False
        if delete_after:
            try:
                os.remove(img_path)
                deleted = True
            except Exception:
                deleted = False
        return jsonify({"message": message, "deleted": deleted})
    except Exception as e:
        return str(e), 400


@app.route("/data/<path:filename>")
def image(filename):
    return send_from_directory(DATA_DIR, filename)


@app.route("/dl/<path:filename>")
def download_bin(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

# ----------------- Mask Secret Image Templates -----------------
MASK_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Mask Image • {{ app_title }}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b1020; color: #e6ebff; }
    header { background:#111733; border-bottom:1px solid #1f2852; padding:12px 20px; }
    header a { color:#9bb5ff; text-decoration:none; font-weight:600; }
    .wrap { max-width: 900px; margin: 40px auto; padding: 24px; }
    .card { background: #111733; border: 1px solid #1f2852; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    label { display:block; font-size:13px; opacity:.9; margin: 10px 0 6px; }
    input[type=file], input[type=password], select { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b376e; background:#0d1330; color:#e6ebff; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .actions { margin-top: 16px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .btn { background:#3355ff; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; }
    .btn.secondary { background:#22306b; }
    .small { font-size: 12px; opacity:.75; }
  </style>
</head>
<body>
  <header>
    <a href="/">← Back to Create</a>
  </header>
  <div class="wrap">
    <div class="card">
      <h2>Mask a secret image with a cover PNG</h2>
      <form action="/mask/create" method="post" enctype="multipart/form-data">
        <label>Cover PNG (visible)</label>
        <input type="file" name="cover" accept="image/png" required>

        <label>Secret image (PNG or JPG)</label>
        <input type="file" name="secret" accept="image/png,image/jpeg" required>

        <div class="row">
          <div>
            <label>Password</label>
            <input type="password" name="password" required>
          </div>
          <div>
            <label>LSBs per channel</label>
            <select name="lsb">
              <option value="1" selected>1 (stealthy)</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </div>
        </div>

        <div class="row">
          <div>
            <label><input type="checkbox" name="use_alpha"> Use alpha channel</label>
          </div>
          <div>
            <label><input type="checkbox" name="no_compress"> Disable compression</label>
          </div>
        </div>

        <div class="actions">
          <button class="btn" type="submit">Create masked link</button>
          <a class="btn secondary" href="/">Cancel</a>
        </div>
      </form>
      <p class="small">Tip: secret images are usually already compressed (JPG/PNG). Compression may not reduce size further.</p>
    </div>
  </div>
</body>
</html>
"""

VIEW_MASK_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Reveal Image • {{ app_title }}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b1020; color: #e6ebff; }
    header { background:#111733; border-bottom:1px solid #1f2852; padding:12px 20px; }
    header a { color:#9bb5ff; text-decoration:none; font-weight:600; margin-right:20px; }
    header a:hover { text-decoration:underline; }
    .wrap { max-width: 1000px; margin: 40px auto; padding: 24px; }
    .card { background: #111733; border: 1px solid #1f2852; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    .imgbox { width: 100%; }
    .imgbox img { width: 100%; height: auto; display: block; border-radius:12px; border:1px solid #243076; }
    .btn { background:#3355ff; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:14px; }
    input[type=password] { flex:1; min-width: 200px; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b376e; background:#0d1330; color:#e6ebff; }
    label.checkbox { display:flex; align-items:center; gap:8px; font-size:13px; opacity:.9; margin-top:8px; }
    .small { font-size: 12px; opacity:.75; }
    .err { color:#ff8a8a; }
  </style>
</head>
<body>
  <header>
    <a href="/">← Back to Create</a>
    <a href="/mask">Mask another</a>
  </header>
  <div class="wrap">
    <div class="card">
      <div class="imgbox"><img id="cover" src="{{ image_url }}" alt="Masked image"></div>
      <div class="row">
        <input id="pwd" type="password" placeholder="Enter password to reveal image">
        <button class="btn" onclick="reveal()">Reveal Image</button>
      </div>
      <label class="checkbox"><input id="delafter" type="checkbox"> Delete this image from server after reveal</label>
      <div id="error" class="small err"></div>
    </div>
  </div>
<script>
async function reveal() {
  const pwd = document.getElementById('pwd').value;
  const del = document.getElementById('delafter').checked;
  document.getElementById('error').textContent = '';
  const res = await fetch(window.location.pathname.replace('/m/','/api/decrypt_image/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: pwd, delete_after: del })
  });
  if (!res.ok) {
    const t = await res.text();
    document.getElementById('error').textContent = t || 'Failed to decrypt.';
    return;
  }
  const data = await res.json();
  if (data.data_url) {
    document.getElementById('cover').src = data.data_url;
  }
  if (data.deleted) {
    const note = document.createElement('div');
    note.className = 'small';
    note.textContent = 'This image has been deleted from the server.';
    document.querySelector('.card').appendChild(note);
  }
}
</script>
</body>
</html>
"""

# ----------------- Mask routes -----------------
@app.route("/mask", methods=["GET"])
def mask_index():
    return render_template_string(MASK_HTML, app_title=APP_TITLE)

@app.route("/mask/create", methods=["POST"])
def mask_create():
    cover = request.files.get("cover")
    secret = request.files.get("secret")
    password = request.form.get("password", "")
    lsb = int(request.form.get("lsb", 1))
    use_alpha = bool(request.form.get("use_alpha"))
    compress = not bool(request.form.get("no_compress"))

    if not cover or not secret or not password:
        return "Missing fields", 400

    tmp_id = secrets.token_urlsafe(8)
    cover_path = os.path.join(DATA_DIR, f"{tmp_id}_cover.png")
    cover.save(cover_path)

    secret_bytes = secret.read()
    # Optional compression (often no benefit for images)
    comp = maybe_compress(secret_bytes, compress)
    encrypted = encrypt_payload(password, comp)

    # Capacity check using cover image
    img = Image.open(cover_path)
    cap = payload_capacity_bytes(img.size, use_alpha, lsb)
    if len(encrypted) > cap:
        os.remove(cover_path)
        return f"Secret image too large for this cover. Capacity ~{cap} bytes; payload {len(encrypted)} bytes.", 400

    sid = secrets.token_urlsafe(10)
    out_path = os.path.join(DATA_DIR, f"{sid}.png")
    embed_into_png(cover_path, out_path, encrypted, lsb, use_alpha, compress)

    try:
        os.remove(cover_path)
    except Exception:
        pass

    return redirect(url_for('view_mask', sid=sid))

@app.route("/m/<sid>")
def view_mask(sid):
    img_path = os.path.join(DATA_DIR, f"{sid}.png")
    if not os.path.exists(img_path):
        abort(404)
    image_url = url_for('image', filename=f"{sid}.png")
    return render_template_string(VIEW_MASK_HTML, image_url=image_url, app_title=APP_TITLE)

@app.route("/api/decrypt_image/<sid>", methods=["POST"]) 
def api_decrypt_image(sid):
    img_path = os.path.join(DATA_DIR, f"{sid}.png")
    if not os.path.exists(img_path):
        return "Not found", 404
    js = request.get_json(silent=True) or {}
    password = js.get("password", "")
    delete_after = bool(js.get("delete_after", False))
    if not password:
        return "Password required", 400
    try:
        encrypted, lsb, use_alpha, compressed = extract_from_png(img_path)
        decrypted = decrypt_payload(password, encrypted)
        if compressed:
            try:
                decrypted = maybe_decompress(decrypted, True)
            except Exception:
                pass
        # Verify it's an image; re-encode to PNG and return data URL
        try:
            im = Image.open(BytesIO(decrypted))
            buf = BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            data_url = f"data:image/png;base64,{b64}"
        except Exception:
            return "Payload is not a valid image.", 400
        # Optionally delete the stego file
        deleted = False
        if delete_after:
            try:
                os.remove(img_path)
                deleted = True
            except Exception:
                deleted = False
        return jsonify({"data_url": data_url, "deleted": deleted})
    except Exception as e:
        return str(e), 400
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route("/admin/cleanup", methods=["POST"]) 
def admin_cleanup():
    # Danger: deletes all PNGs and extracted bins in data dir
    if not os.path.isdir(DATA_DIR):
        return redirect(url_for('index'))
    removed = 0
    for name in os.listdir(DATA_DIR):
        if name.endswith('.png') or name.endswith('_extracted.bin'):
            try:
                os.remove(os.path.join(DATA_DIR, name))
                removed += 1
            except Exception:
                pass
    return f"Deleted {removed} files from data/ — <a href='/' style='color:#9bb5ff'>Back</a>", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
