# 鸟类标注分享工具 Web App (MVP)

## 0. 准备环境（macOS + VS Code + Copilot）

``` bash
mkdir bird-captioner && cd bird-captioner
python3 -m venv .venv
source .venv/bin/activate
pip install flask pillow pydantic sqlite-utils pypinyin python-dotenv
```

VS Code 设置： - `Cmd+Shift+P` → "Python: Select Interpreter" → 选
`.venv` - 安装扩展：Python、Pylance、GitHub Copilot - 初始化
Git：`git init`，并添加 `.gitignore`：

    .venv/
    __pycache__/
    instance/
    uploads/
    outputs/
    .env

------------------------------------------------------------------------

## 1. 项目结构

    bird-captioner/
      app.py
      requirements.txt
      .env
      /static/
      /templates/
      /fonts/
      /data/
      /uploads/
      /outputs/
      seeds/species.csv

------------------------------------------------------------------------

## 2. 初始化数据

`seeds/species.csv`

``` csv
chinese_name,latin_name
白头鹎,Pycnonotus sinensis
黑水鸡,Gallinula chloropus
小白鹭,Egretta garzetta
```

`scripts/init_db.py`

``` python
import csv, sqlite3
from pypinyin import lazy_pinyin

con = sqlite3.connect("data/species.sqlite")
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS species(
  id INTEGER PRIMARY KEY,
  chinese_name TEXT NOT NULL,
  latin_name TEXT NOT NULL,
  pinyin TEXT
)""")
cur.execute("DELETE FROM species")
with open("seeds/species.csv", newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        cn = row["chinese_name"].strip()
        la = row["latin_name"].strip()
        py = "".join(lazy_pinyin(cn))
        cur.execute("INSERT INTO species(chinese_name, latin_name, pinyin) VALUES (?, ?, ?)", (cn, la, py))
con.commit()
con.close()
```

运行：

``` bash
python scripts/init_db.py
```

------------------------------------------------------------------------

## 3. Flask 应用 (app.py)

``` python
from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os, sqlite3

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["DB_PATH"] = "data/species.sqlite"
app.config["FONT_PATH"] = "fonts/NotoSansCJK-Regular.ttf"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

def db_conn():
    return sqlite3.connect(app.config["DB_PATH"])

def search_species(q: str, limit: int = 10):
    q = q.strip()
    if not q:
        return []
    sql = """
      SELECT chinese_name, latin_name
      FROM species
      WHERE chinese_name LIKE ? OR latin_name LIKE ? OR pinyin LIKE ?
      LIMIT ?
    """
    like = f"%{q}%"
    with db_conn() as con:
        return [{"chinese_name": r[0], "latin_name": r[1]} 
                for r in con.execute(sql, (like, like, like, limit)).fetchall()]

def overlay_caption(img_path: str, cn: str, la: str,
                    position: str = "bottom_right",
                    margin: int = 24, alpha_bg: int = 128):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    base = max(18, int(W * 0.018))
    font_cn = ImageFont.truetype(app.config["FONT_PATH"], base)
    font_la = ImageFont.truetype(app.config["FONT_PATH"], int(base*0.9))

    line_cn = cn
    line_la = la
    w_cn, h_cn = draw.textbbox((0,0), line_cn, font=font_cn)[2:]
    w_la, h_la = draw.textbbox((0,0), line_la, font=font_la)[2:]

    pad_x, pad_y = 16, 12
    box_w = max(w_cn, w_la) + pad_x*2
    box_h = (h_cn + h_la) + pad_y*3

    if position == "bottom_right":
        x0 = W - box_w - margin
        y0 = H - box_h - margin
    elif position == "bottom_left":
        x0 = margin
        y0 = H - box_h - margin
    else:
        x0 = W - box_w - margin
        y0 = H - box_h - margin

    draw.rectangle([x0, y0, x0+box_w, y0+box_h], fill=(0,0,0,alpha_bg), outline=None)
    draw.text((x0+pad_x, y0+pad_y), line_cn, font=font_cn, fill=(255,255,255,255))
    draw.text((x0+pad_x, y0+pad_y+h_cn+6), line_la, font=font_la, fill=(255,255,255,230))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], f"captioned_{ts}.jpg")
    img.save(out_path, quality=95)
    return out_path

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify(search_species(q))

@app.post("/upload")
def upload():
    f = request.files.get("photo")
    cn = request.form.get("chinese_name", "").strip()
    la = request.form.get("latin_name", "").strip()
    pos = request.form.get("position", "bottom_right")
    if not (f and cn and la):
        return "Missing fields", 400
    path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(path)
    out = overlay_caption(path, cn, la, position=pos)
    return render_template("preview.html", out_path=out)

@app.get("/download")
def download():
    p = request.args.get("p")
    if not p or not os.path.exists(p):
        return "Not found", 404
    return send_file(p, as_attachment=True)
    
if __name__ == "__main__":
    app.run(debug=True)
```

------------------------------------------------------------------------

## 4. 模板

### base.html

``` html
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>鸟类标注分享工具</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
</head>
<body class="container">
  <header><h2>鸟类标注分享工具（MVP）</h2></header>
  <main>{% block content %}{% endblock %}</main>
  <footer><small>© 2025</small></footer>
</body>
</html>
```

### index.html

``` html
{% extends "base.html" %}
{% block content %}
<form action="/upload" method="post" enctype="multipart/form-data">
  <label>上传照片<input type="file" name="photo" accept="image/*" required></label>
  <label>中文名<input id="cn" name="chinese_name" required></label>
  <label>拉丁学名<input id="la" name="latin_name" required></label>
  <label>位置<select name="position"><option value="bottom_right">右下</option><option value="bottom_left">左下</option></select></label>
  <button type="submit">生成</button>
</form>
{% endblock %}
```

### preview.html

``` html
{% extends "base.html" %}
{% block content %}
<img src="/{{ out_path }}" style="max-width:100%;height:auto;"/>
<p><a href="/download?p={{ out_path }}">下载</a></p>
{% endblock %}
```

------------------------------------------------------------------------

## 5. 运行

``` bash
flask --app app.py run
```

访问 `http://127.0.0.1:5000`

------------------------------------------------------------------------

## 6. 下一步扩展

-   支持搜索接口自动填充（拼音/拉丁学名）。
-   批量处理。
-   自动识别候选（接入 Merlin Bird ID 等模型）。
-   社交媒体一键分享。
