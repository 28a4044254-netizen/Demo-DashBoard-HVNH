"""
HỆ THỐNG PHÂN TÍCH & HỖ TRỢ HỌC VỤ - HỌC VIỆN NGÂN HÀNG (phiên bản 2)

Chạy:  streamlit run app.py

Vai trò:
  - admin   : CVHT quản trị - xem tất cả, nạp dữ liệu, tạo/quản lý tài khoản
  - advisor : CVHT/giảng viên - xem các lớp được phân công
  - student : sinh viên - chỉ xem hồ sơ của chính mình
"""
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import time
import unicodedata
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

st.set_page_config(
    page_title="Hệ thống Phân tích & Hỗ trợ Học vụ - HVNH",
    page_icon="🎓",
    layout="wide",
)

# =========================================================
# 0. CẤU HÌNH TẬP TRUNG (mọi ngưỡng chỉ khai báo MỘT lần ở đây)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("HVNH_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "users.db"
DATASET_PATH = DATA_DIR / "dataset.csv"
META_PATH = DATA_DIR / "dataset_meta.json"

TH = {
    "gpa_critical": 1.2,        # GPA suy giảm nghiêm trọng
    "gpa_low": 1.8,             # GPA thấp
    "gpa_overload": 2.0,        # GPA dưới mức này mà đăng ký quá tải thì cảnh báo
    "credits_overload": 21,     # số TC đăng ký bị coi là quá tải
    "debt_watch": 6,            # có nợ tín chỉ đáng chú ý
    "debt_high": 9,             # nợ đọng nhiều
    "absence_watch": 12,        # chuyên cần giảm (%)
    "absence_high": 20,         # vắng quá mức (%)
    "diligent_absence": 15,     # "chăm học": vắng dưới mức này (%)
    "level_red": 70,            # điểm rủi ro >= -> Đỏ
    "level_yellow": 40,         # điểm rủi ro >= -> Vàng
}
WEIGHTS = {"gpa": 16, "debt": 2.8, "absence": 0.85, "overload": 8}

RED, YELLOW, GREEN = "Nguy cơ cao (Đỏ)", "Cần theo dõi (Vàng)", "An toàn (Xanh)"
STUDENT_LABEL = {RED: "🔴 Cần được hỗ trợ sớm", YELLOW: "🟡 Cần lưu ý", GREEN: "🟢 Ổn định"}

FEATURES = ["GPA_Ky_Truoc", "GPA_Tich_Luy", "Tin_Chi_Dang_Ky",
            "Tin_Chi_No", "Ty_Le_Vang_Mat", "Diem_Ren_Luyen"]
LABEL_COL = "Bi_Canh_Bao_Ky_Sau"   # nhãn thật (0/1) - nếu có sẽ huấn luyện mô hình ML

# tên chuẩn: (mô tả, bắt buộc?, các tên gọi thường gặp)
COLUMN_SPEC = {
    "Ma_SV": ("Mã sinh viên", True, ["masv", "mssv", "masinhvien", "studentid", "id"]),
    "Ho_Ten": ("Họ và tên", False, ["hoten", "hovaten", "ten", "name", "fullname"]),
    "Lop_Nien_Che": ("Lớp niên chế", False, ["lopnienche", "lop", "class", "lopsinhhoat"]),
    "GPA_Ky_Truoc": ("GPA kỳ trước", True, ["gpakytruoc", "gpahocky", "gpahk", "dtbhocky", "gpa"]),
    "GPA_Tich_Luy": ("GPA tích lũy (CPA)", False, ["gpatichluy", "cpa", "gpatl", "dtbtichluy"]),
    "Tin_Chi_Dang_Ky": ("Số TC đăng ký kỳ này", False, ["tinchidangky", "tcdk", "sotinchidangky"]),
    "Tin_Chi_No": ("Số TC nợ", True, ["tinchino", "tcno", "sotinchino", "notinchi"]),
    "Ty_Le_Vang_Mat": ("Tỷ lệ vắng mặt (%)", True, ["tylevangmat", "vangmat", "tylevang", "absence"]),
    "Diem_Ren_Luyen": ("Điểm rèn luyện", False, ["diemrenluyen", "drl"]),
    LABEL_COL: ("Nhãn thật: bị cảnh báo kỳ sau (0/1)", False,
                ["bicanhbaokysau", "canhbaokysau", "canhbao", "label"]),
}
NUMERIC_COLS = ["GPA_Ky_Truoc", "GPA_Tich_Luy", "Tin_Chi_Dang_Ky", "Tin_Chi_No",
                "Ty_Le_Vang_Mat", "Diem_Ren_Luyen", LABEL_COL]
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
_NEW_WIDTH_API = _ST_VER >= (1, 50)


# =========================================================
# 1. TIỆN ÍCH GIAO DIỆN
# =========================================================
def show_df(data, **kw):
    if _NEW_WIDTH_API:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


def show_chart(fig):
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def color_status(val):
    if val == RED:
        return "background-color: #ff4b4b; color: white; font-weight: bold"
    if val == YELLOW:
        return "background-color: #ffa421; color: black; font-weight: bold"
    return "background-color: #21c354; color: white"


def style_status(df, col="Trang_Thai"):
    sty = df.style
    fn = sty.map if hasattr(sty, "map") else sty.applymap  # pandas >= 2.1 dùng .map
    return fn(color_status, subset=[col])


# =========================================================
# 2. CƠ SỞ DỮ LIỆU TÀI KHOẢN & XÁC THỰC
# =========================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_exec(sql, params=()):
    with closing(get_conn()) as c:
        with c:
            c.execute(sql, params)


def db_query(sql, params=()):
    with closing(get_conn()) as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def init_db():
    db_exec("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin','advisor','student')),
            full_name TEXT DEFAULT '',
            classes TEXT DEFAULT '',
            student_id TEXT DEFAULT '',
            must_change INTEGER DEFAULT 0,
            failed INTEGER DEFAULT 0,
            locked_until REAL DEFAULT 0,
            created_at TEXT
        )""")


def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               bytes.fromhex(salt_hex), 200_000).hex()


def password_error(pw):
    if len(pw) < 8:
        return "Mật khẩu phải có ít nhất 8 ký tự."
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return "Mật khẩu phải gồm cả chữ và số."
    return None


def create_user(username, password, role, full_name="", classes="",
                student_id="", must_change=0):
    salt = secrets.token_hex(16)
    db_exec(
        "INSERT INTO users (username,password_hash,salt,role,full_name,classes,"
        "student_id,must_change,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (username.strip().lower(), hash_password(password, salt), salt, role,
         full_name, classes, student_id, must_change,
         datetime.now().strftime("%Y-%m-%d %H:%M")),
    )


def get_user(username):
    rows = db_query("SELECT * FROM users WHERE username=?", ((username or "").strip().lower(),))
    return rows[0] if rows else None


def set_password(username, new_password, must_change=0):
    salt = secrets.token_hex(16)
    db_exec(
        "UPDATE users SET password_hash=?, salt=?, must_change=?, failed=0, locked_until=0 "
        "WHERE username=?",
        (hash_password(new_password, salt), salt, must_change, username),
    )


MAX_FAILED, LOCK_SECONDS = 5, 300


def authenticate(username, password):
    """Trả về (user | None, thông báo lỗi)."""
    user = get_user(username)
    if user is None:
        hash_password(password, "00" * 16)  # cân bằng thời gian phản hồi
        return None, "Sai tên đăng nhập hoặc mật khẩu."
    if user["locked_until"] and user["locked_until"] > time.time():
        mins = int((user["locked_until"] - time.time()) // 60) + 1
        return None, f"Tài khoản tạm khóa do nhập sai nhiều lần. Thử lại sau ~{mins} phút."
    ok = hmac.compare_digest(hash_password(password, user["salt"]), user["password_hash"])
    if not ok:
        failed = user["failed"] + 1
        locked = time.time() + LOCK_SECONDS if failed >= MAX_FAILED else 0
        db_exec("UPDATE users SET failed=?, locked_until=? WHERE username=?",
                (0 if locked else failed, locked, user["username"]))
        return None, "Sai tên đăng nhập hoặc mật khẩu."
    db_exec("UPDATE users SET failed=0, locked_until=0 WHERE username=?", (user["username"],))
    return user, ""


def create_student_accounts(df):
    """Tạo tài khoản cho SV chưa có; trả về DataFrame thông tin đăng nhập ban đầu."""
    existing = {u["username"] for u in db_query("SELECT username FROM users")}
    out = []
    with closing(get_conn()) as c:
        with c:
            for ma, ten in df[["Ma_SV", "Ho_Ten"]].itertuples(index=False):
                uname = str(ma).strip().lower()
                if uname in existing:
                    continue
                pw = secrets.token_urlsafe(6)
                salt = secrets.token_hex(16)
                c.execute(
                    "INSERT INTO users (username,password_hash,salt,role,full_name,classes,"
                    "student_id,must_change,created_at) VALUES (?,?,?,'student',?,'',?,1,?)",
                    (uname, hash_password(pw, salt), salt, ten, ma,
                     datetime.now().strftime("%Y-%m-%d %H:%M")),
                )
                out.append({"Tên đăng nhập": uname, "Họ tên": ten, "Mật khẩu tạm": pw})
    return pd.DataFrame(out)


# =========================================================
# 3. CHUẨN HÓA & KIỂM TRA DỮ LIỆU ĐẦU VÀO
# =========================================================
def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s).replace("đ", "d").replace("Đ", "D"))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def auto_map(columns):
    """Tự đoán ánh xạ cột: tên chuẩn -> tên cột trong file (hoặc None)."""
    normed = {c: norm_name(c) for c in columns}
    used, mapping = set(), {}
    for std, (_, _, aliases) in COLUMN_SPEC.items():
        targets = [norm_name(std)] + aliases
        found = None
        for t in targets:            # ưu tiên theo thứ tự alias
            for col, nc in normed.items():
                if nc == t and col not in used:
                    found = col
                    break
            if found:
                break
        mapping[std] = found
        if found:
            used.add(found)
    return mapping


def standardize(raw, mapping):
    """Trả về (df_chuẩn | None, danh_sách_lỗi, danh_sách_cảnh_báo)."""
    errors, warns = [], []
    missing = [COLUMN_SPEC[c][0] for c, (_, req, _) in COLUMN_SPEC.items()
               if req and not mapping.get(c)]
    if missing:
        return None, ["Thiếu cột bắt buộc: " + ", ".join(missing)], warns

    df = pd.DataFrame({std: raw[src].to_numpy() for std, src in mapping.items() if src})
    n0 = len(df)

    df["Ma_SV"] = df["Ma_SV"].astype(str).str.strip()
    df = df[~df["Ma_SV"].isin(["", "nan", "None", "NaN"])]

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce")

    req_num = [c for c in ["GPA_Ky_Truoc", "Tin_Chi_No", "Ty_Le_Vang_Mat"]]
    bad = df[req_num].isna().any(axis=1)
    if bad.sum():
        warns.append(f"Đã loại {int(bad.sum())} dòng thiếu/sai định dạng ở cột bắt buộc "
                     "(GPA kỳ trước, TC nợ, tỷ lệ vắng).")
        df = df[~bad]

    dup = df["Ma_SV"].duplicated()
    if dup.sum():
        warns.append(f"Đã loại {int(dup.sum())} dòng trùng Mã SV (giữ dòng đầu tiên).")
        df = df[~dup]

    # Chặn giá trị ngoài miền hợp lệ
    ranges = {"GPA_Ky_Truoc": (0, 4), "GPA_Tich_Luy": (0, 4), "Tin_Chi_No": (0, 200),
              "Ty_Le_Vang_Mat": (0, 100), "Diem_Ren_Luyen": (0, 100), "Tin_Chi_Dang_Ky": (0, 40)}
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            out = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            if out.sum():
                warns.append(f"{int(out.sum())} giá trị '{col}' ngoài khoảng [{lo}, {hi}] "
                             "đã được cắt về biên.")
                df[col] = df[col].clip(lo, hi)

    # Bù cột tùy chọn
    if "Ho_Ten" not in df.columns:
        df["Ho_Ten"] = df["Ma_SV"]
        warns.append("Không có cột Họ tên: dùng Mã SV thay thế.")
    df["Ho_Ten"] = df["Ho_Ten"].fillna(df["Ma_SV"]).astype(str)
    if "Lop_Nien_Che" not in df.columns:
        df["Lop_Nien_Che"] = "Chưa phân lớp"
        warns.append("Không có cột Lớp niên chế: gán 'Chưa phân lớp'.")
    df["Lop_Nien_Che"] = df["Lop_Nien_Che"].fillna("Chưa phân lớp").astype(str)
    if "GPA_Tich_Luy" not in df.columns:
        df["GPA_Tich_Luy"] = df["GPA_Ky_Truoc"]
        warns.append("Không có GPA tích lũy: dùng GPA kỳ trước thay thế.")
    df["GPA_Tich_Luy"] = df["GPA_Tich_Luy"].fillna(df["GPA_Ky_Truoc"])
    if "Tin_Chi_Dang_Ky" not in df.columns:
        df["Tin_Chi_Dang_Ky"] = 15
        warns.append("Không có số TC đăng ký: giả định 15 TC (ảnh hưởng cảnh báo quá tải).")
    df["Tin_Chi_Dang_Ky"] = df["Tin_Chi_Dang_Ky"].fillna(15)
    if "Diem_Ren_Luyen" not in df.columns:
        df["Diem_Ren_Luyen"] = np.nan
    df["Diem_Ren_Luyen"] = df["Diem_Ren_Luyen"].fillna(
        df["Diem_Ren_Luyen"].median() if df["Diem_Ren_Luyen"].notna().any() else 75)

    if LABEL_COL in df.columns and df[LABEL_COL].notna().sum() == 0:
        df = df.drop(columns=[LABEL_COL])
        warns.append("Cột nhãn không có giá trị 0/1 hợp lệ: bỏ qua.")

    if df.empty:
        return None, ["Không còn dòng dữ liệu hợp lệ sau khi kiểm tra."], warns
    if len(df) < n0:
        warns.append(f"Còn {len(df)}/{n0} dòng hợp lệ.")
    return df.reset_index(drop=True), errors, warns


def generate_sample_dataset(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Ma_SV": [f"SV28{str(i).zfill(4)}" for i in range(1, n + 1)],
        "Ho_Ten": [f"Sinh viên {i}" for i in range(1, n + 1)],
        "Lop_Nien_Che": rng.choice(["K28-HTTTQL", "K28-NH", "K27-TC", "K26-KT"], n),
        "GPA_Ky_Truoc": np.round(rng.uniform(0.7, 3.8, n), 2),
        "GPA_Tich_Luy": np.round(rng.uniform(0.9, 3.7, n), 2),
        "Tin_Chi_Dang_Ky": rng.choice([12, 15, 18, 21, 24], n),
        "Tin_Chi_No": rng.choice([0, 0, 0, 3, 6, 9, 12, 15], n),
        "Ty_Le_Vang_Mat": rng.integers(0, 38, n),
        "Diem_Ren_Luyen": rng.integers(48, 100, n),
    })


# =========================================================
# 4. LƯU / TẢI DỮ LIỆU DÙNG CHUNG (CVHT nạp - SV xem)
# =========================================================
def save_dataset(df, source_name, uploader):
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    META_PATH.write_text(json.dumps({
        "source": source_name, "uploaded_by": uploader, "rows": len(df),
        "uploaded_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }, ensure_ascii=False), encoding="utf-8")


def dataset_mtime():
    return DATASET_PATH.stat().st_mtime if DATASET_PATH.exists() else None


def load_meta():
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_dataset(mtime):
    return pd.read_csv(DATASET_PATH, encoding="utf-8-sig", dtype={"Ma_SV": str})


@st.cache_resource(show_spinner=False)
def get_model(mtime):
    """Chỉ huấn luyện ML khi file có cột nhãn THẬT. Trả về (model, thông tin)."""
    if not HAS_SKLEARN or mtime is None:
        return None, None
    df = load_dataset(mtime)
    if LABEL_COL not in df.columns:
        return None, None
    d = df.dropna(subset=[LABEL_COL])
    y = d[LABEL_COL].astype(int)
    if len(d) < 100 or y.nunique() < 2 or y.value_counts().min() < 15:
        return None, {"note": "Có cột nhãn nhưng chưa đủ dữ liệu (cần ≥100 dòng, "
                              "mỗi lớp ≥15) để huấn luyện mô hình tin cậy."}
    X_tr, X_te, y_tr, y_te = train_test_split(d[FEATURES], y, test_size=0.25,
                                              stratify=y, random_state=42)
    params = dict(n_estimators=200, max_depth=5, min_samples_leaf=3,
                  class_weight="balanced", random_state=42)
    auc = roc_auc_score(y_te, RandomForestClassifier(**params).fit(X_tr, y_tr)
                        .predict_proba(X_te)[:, 1])
    final = RandomForestClassifier(**params).fit(d[FEATURES], y)
    return final, {"auc": float(auc), "n": int(len(d))}


# =========================================================
# 5. BỘ MÁY CHẤM ĐIỂM RỦI RO & CHẨN ĐOÁN (vector hóa)
# =========================================================
def _join(sep, *cols):
    return [sep.join(x for x in t if x) for t in zip(*cols)]


def analyze(df_in, model=None):
    df = df_in.copy()
    gpa, no = df["GPA_Ky_Truoc"], df["Tin_Chi_No"]
    vang, dk = df["Ty_Le_Vang_Mat"], df["Tin_Chi_Dang_Ky"]

    overload = (dk >= TH["credits_overload"]) & (gpa < TH["gpa_overload"])
    rule_score = ((4.0 - gpa) * WEIGHTS["gpa"] + no * WEIGHTS["debt"]
                  + vang * WEIGHTS["absence"] + overload * WEIGHTS["overload"])
    if model is not None and all(c in df.columns for c in FEATURES):
        score = model.predict_proba(df[FEATURES])[:, 1] * 100
        df["Nguon_Diem"] = "Mô hình ML (huấn luyện trên nhãn thực)"
    else:
        score = rule_score
        df["Nguon_Diem"] = "Chấm điểm theo quy tắc"
    df["Diem_Rui_Ro"] = np.clip(np.round(score, 1), 0, 100)

    df["Trang_Thai"] = np.select(
        [df["Diem_Rui_Ro"] >= TH["level_red"], df["Diem_Rui_Ro"] >= TH["level_yellow"]],
        [RED, YELLOW], default=GREEN)

    ni = no.round().astype(int).astype(str)
    vi = vang.round().astype(int).astype(str)
    di = dk.round().astype(int).astype(str)

    def pick(conds, choices):
        return np.select(conds, choices, default="")

    c_g = [gpa < TH["gpa_critical"], gpa < TH["gpa_low"]]
    c_d = [no >= TH["debt_high"], no >= TH["debt_watch"]]
    c_v = [vang >= TH["absence_high"], vang >= TH["absence_watch"]]
    c_o = [overload]

    reasons = _join(
        " + ", 
        pick(c_g, [f"GPA suy giảm nghiêm trọng (<{TH['gpa_critical']})",
                   f"GPA học kỳ thấp (<{TH['gpa_low']})"]),
        pick(c_d, [("Nợ đọng nhiều tín chỉ (" + ni + " TC)").to_numpy(),
                   ("Có tín chỉ nợ (" + ni + " TC)").to_numpy()]),
        pick(c_v, [("Vắng học nhiều (" + vi + "%)").to_numpy(),
                   ("Chuyên cần giảm (" + vi + "%)").to_numpy()]),
        pick(c_o, [("Đăng ký quá tải (" + di + " TC) so với kết quả hiện tại").to_numpy()]),
    )
    adv = _join(
        "; ",
        pick(c_g, ["Mời gặp trực tiếp; rà soát phương pháp học và ôn thi",
                   "Ghép nhóm Đôi bạn cùng tiến hỗ trợ học tập"]),
        pick(c_d, ["Đề nghị giảm môn mới (tối đa 12-14 TC) để ưu tiên trả nợ môn",
                   "Ưu tiên đăng ký học lại học phần tiên quyết"]),
        pick(c_v, ["Nhắc nhở chuyên cần; tìm hiểu lý do (sức khỏe, đi làm thêm...)",
                   "Nhắc nhở đi học đầy đủ để giữ điểm chuyên cần"]),
        pick(c_o, ["Xem xét giảm số tín chỉ đăng ký kỳ tới"]),
    )
    stu = _join(
        "; ",
        pick(c_g, ["Đặt lịch gặp CVHT sớm để cùng rà soát phương pháp học",
                   "Tham gia nhóm học tập/Đôi bạn cùng tiến để được hỗ trợ"]),
        pick(c_d, ["Cân nhắc đăng ký ít môn mới (khoảng 12-14 TC) để ưu tiên học lại môn nợ",
                   "Ưu tiên đăng ký học lại các học phần tiên quyết"]),
        pick(c_v, ["Cố gắng đi học đều hơn; nếu có khó khăn hãy chia sẻ với CVHT",
                   "Duy trì đi học đầy đủ để giữ điểm chuyên cần"]),
        pick(c_o, ["Cân nhắc giảm số tín chỉ đăng ký kỳ tới"]),
    )

    green = (df["Trang_Thai"] == GREEN).to_numpy()
    df["Nguyen_Nhan_Chinh"] = [
        r if r else ("Tiến độ học tập ổn định" if g else "Điểm tổng hợp ở mức cần chú ý")
        for r, g in zip(reasons, green)]
    df["De_Xuat_Cho_CVHT"] = [
        a if a else ("Duy trì lộ trình chuẩn; khuyến khích NCKH/học bổng" if g
                     else "Theo dõi định kỳ giữa học kỳ")
        for a, g in zip(adv, green)]
    df["Goi_Y_Cho_SV"] = [
        s if s else ("Bạn đang học tốt - hãy duy trì và cân nhắc NCKH/học bổng" if g
                     else "Tiếp tục theo dõi tiến độ của mình")
        for s, g in zip(stu, green)]
    return df


@st.cache_data(show_spinner=False)
def analyzed_dataset(mtime, _model):
    return analyze(load_dataset(mtime), _model)


# =========================================================
# 6. CÁC MÀN HÌNH ĐĂNG NHẬP
# =========================================================
def brand_header():
    if (BASE_DIR / "logo.png").exists():
        st.sidebar.image(str(BASE_DIR / "logo.png"), width=160)
    else:
        st.sidebar.markdown(
            """<div style="background-color:#003366;padding:12px;border-radius:8px;
            text-align:center;border:1px solid #4da6ff;">
            <h3 style="color:white;margin:0;font-size:17px;">🏦 HỌC VIỆN NGÂN HÀNG</h3>
            <p style="color:#b3d9ff;margin:4px 0 0 0;font-size:11px;">
            HỆ THỐNG PHÂN TÍCH & HỖ TRỢ HỌC VỤ</p></div>""",
            unsafe_allow_html=True)


def setup_screen():
    st.title("🔧 Thiết lập lần đầu")
    st.write("Chưa có tài khoản nào. Hãy tạo **tài khoản quản trị (CVHT)** đầu tiên.")
    with st.form("setup"):
        u = st.text_input("Tên đăng nhập")
        name = st.text_input("Họ tên")
        p1 = st.text_input("Mật khẩu", type="password")
        p2 = st.text_input("Nhập lại mật khẩu", type="password")
        if st.form_submit_button("Tạo tài khoản quản trị"):
            if not u.strip():
                st.error("Vui lòng nhập tên đăng nhập.")
            elif password_error(p1):
                st.error(password_error(p1))
            elif p1 != p2:
                st.error("Mật khẩu nhập lại không khớp.")
            else:
                create_user(u, p1, "admin", full_name=name)
                st.success("Đã tạo tài khoản. Hãy đăng nhập.")
                time.sleep(1)
                st.rerun()


def login_screen():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## 🎓 Đăng nhập hệ thống")
        st.caption("Dành cho Cố vấn học tập / Giảng viên và Sinh viên Học viện Ngân hàng.")
        with st.form("login"):
            u = st.text_input("Tên đăng nhập (sinh viên dùng Mã SV)")
            p = st.text_input("Mật khẩu", type="password")
            if st.form_submit_button("Đăng nhập", type="primary"):
                user, msg = authenticate(u, p)
                if user:
                    st.session_state["uname"] = user["username"]
                    st.rerun()
                else:
                    st.error(msg)
        st.caption("Quên mật khẩu? Liên hệ Cố vấn học tập để được cấp lại.")


def password_form(user, forced=False):
    with st.form(f"pw_{'forced' if forced else 'normal'}"):
        old = st.text_input("Mật khẩu hiện tại", type="password")
        n1 = st.text_input("Mật khẩu mới", type="password")
        n2 = st.text_input("Nhập lại mật khẩu mới", type="password")
        if st.form_submit_button("Đổi mật khẩu"):
            cur = get_user(user["username"])
            if not hmac.compare_digest(hash_password(old, cur["salt"]), cur["password_hash"]):
                st.error("Mật khẩu hiện tại không đúng.")
            elif password_error(n1):
                st.error(password_error(n1))
            elif n1 != n2:
                st.error("Mật khẩu nhập lại không khớp.")
            elif n1 == old:
                st.error("Mật khẩu mới phải khác mật khẩu cũ.")
            else:
                set_password(user["username"], n1, must_change=0)
                st.success("Đã đổi mật khẩu.")
                time.sleep(1)
                st.rerun()


def force_change_screen(user):
    st.title("🔐 Đổi mật khẩu lần đầu")
    st.info("Đây là mật khẩu tạm do CVHT cấp. Hãy đặt mật khẩu riêng trước khi tiếp tục.")
    password_form(user, forced=True)
    if st.button("Đăng xuất"):
        st.session_state.pop("uname", None)
        st.rerun()


# =========================================================
# 7. CÁC TRANG DÀNH CHO CVHT / GIẢNG VIÊN
# =========================================================
def scope_df(df, user):
    if user["role"] == "admin" or not user["classes"].strip():
        return df
    allowed = [c.strip() for c in user["classes"].split(",") if c.strip()]
    return df[df["Lop_Nien_Che"].isin(allowed)]


def page_overview(df, lop_chon, model_info, source_text):
    n = len(df)
    sv_do = df[df["Trang_Thai"] == RED]
    sv_vang = df[df["Trang_Thai"] == YELLOW]
    so_do, so_vang = len(sv_do), len(sv_vang)
    ty_le_do = round(so_do / n * 100, 1)
    so_no = int((df["Tin_Chi_No"] >= TH["debt_high"]).sum())
    so_vang_nhieu = int((df["Ty_Le_Vang_Mat"] >= TH["absence_high"]).sum())
    so_gpa_kem = int((df["GPA_Ky_Truoc"] < 1.5).sum())

    st.title("📊 Báo cáo Phân tích Dữ liệu Học vụ")
    st.caption(f"Nguồn dữ liệu: {source_text}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Quy mô sinh viên", f"{n} SV")
    c2.metric("🚨 Nguy cơ cao (Đỏ)", f"{so_do} SV ({ty_le_do}%)")
    c3.metric("⚠️ Cần theo dõi (Vàng)", f"{so_vang} SV")
    c4.metric("📈 GPA kỳ trước trung bình", f"{df['GPA_Ky_Truoc'].mean():.2f} / 4.0")

    if model_info and "auc" in model_info:
        st.success(f"Điểm rủi ro đang dùng **mô hình ML** huấn luyện trên {model_info['n']} "
                   f"sinh viên có nhãn thực; AUC kiểm tra = **{model_info['auc']:.2f}**.")
    else:
        extra = f" {model_info['note']}" if model_info and "note" in model_info else ""
        st.caption("ℹ️ Điểm rủi ro được **chấm theo quy tắc** (công thức trọng số, minh bạch). "
                   f"Để dùng ML, thêm cột nhãn thật '{LABEL_COL}' (0/1) khi nạp dữ liệu.{extra}")

    st.markdown("### 🤖 Nhận định tự động từ dữ liệu")
    st.info("\n".join([
        f"**Tình hình lớp ({lop_chon}):** trong **{n}** sinh viên, có **{so_do} SV ({ty_le_do}%)** "
        f"ở mức Đỏ và **{so_vang} SV** ở mức Vàng.",
        "",
        f"- **Nợ tín chỉ:** {so_no} SV nợ từ {TH['debt_high']} TC trở lên "
        f"({round(so_no / n * 100, 1)}%), dễ quá tải nếu đăng ký thêm môn mới.",
        f"- **Chuyên cần:** {so_vang_nhieu} SV vắng từ {TH['absence_high']}% trở lên, "
        "có nguy cơ cấm thi/điểm quá trình thấp.",
        f"- **Điểm học kỳ:** {so_gpa_kem} SV có GPA kỳ trước dưới 1.5/4.0.",
        "",
        f"**Đề xuất trọng tâm:** họp rà soát kế hoạch đăng ký tín chỉ cho {so_do} SV nhóm Đỏ; "
        f"yêu cầu nhóm nợ ≥ {TH['debt_high']} TC rút bớt học phần tự chọn chưa cấp thiết.",
    ]))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Cơ cấu nhóm rủi ro")
        show_chart(px.pie(df, names="Trang_Thai", hole=0.38, color="Trang_Thai",
                          color_discrete_map={RED: "#ff4b4b", YELLOW: "#ffa421", GREEN: "#21c354"}))
    with col2:
        st.subheader("2. Các nguyên nhân rủi ro hàng đầu")
        causes = pd.DataFrame({
            "Nguyên nhân": [f"GPA thấp (< 1.5)", f"Nợ tín chỉ (≥ {TH['debt_watch']} TC)",
                            f"Vắng học nhiều (≥ {TH['absence_high']}%)",
                            f"Đăng ký quá tải (≥ {TH['credits_overload']} TC & GPA < {TH['gpa_overload']})"],
            "Số SV": [so_gpa_kem, int((df["Tin_Chi_No"] >= TH["debt_watch"]).sum()), so_vang_nhieu,
                      int(((df["Tin_Chi_Dang_Ky"] >= TH["credits_overload"])
                           & (df["GPA_Ky_Truoc"] < TH["gpa_overload"])).sum())],
        })
        fig = px.bar(causes, x="Số SV", y="Nguyên nhân", orientation="h", text="Số SV",
                     color="Số SV", color_continuous_scale="Reds")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(fig)

    st.subheader("📋 Bảng chi tiết kèm chẩn đoán & đề xuất")
    sel = st.multiselect("Lọc theo trạng thái:", [RED, YELLOW, GREEN], default=[RED, YELLOW])
    cols = ["Ma_SV", "Ho_Ten", "Lop_Nien_Che", "GPA_Ky_Truoc", "Tin_Chi_No", "Ty_Le_Vang_Mat",
            "Diem_Rui_Ro", "Trang_Thai", "Nguyen_Nhan_Chinh", "De_Xuat_Cho_CVHT"]
    view = df[df["Trang_Thai"].isin(sel)].sort_values("Diem_Rui_Ro", ascending=False)[cols]
    show_df(style_status(view), height=380)
    st.download_button("📥 Xuất báo cáo đã phân tích (.CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       "Bao_Cao_Phan_Tich_CVHT_HVNH.csv", "text/csv")


def email_text(r):
    return f"""Kính gửi em {r['Ho_Ten']} (Mã SV: {r['Ma_SV']} - Lớp {r['Lop_Nien_Che']}),

Qua theo dõi dữ liệu học tập đầu kỳ, Cố vấn học tập muốn trao đổi với em để cùng chuẩn bị tốt cho học kỳ tới.

- Kết quả ghi nhận: GPA kỳ trước {r['GPA_Ky_Truoc']}/4.0; số tín chỉ nợ: {int(r['Tin_Chi_No'])}; tỷ lệ vắng mặt: {int(r['Ty_Le_Vang_Mat'])}%.
- Điểm cần lưu ý: {r['Nguyen_Nhan_Chinh']}.
- Gợi ý định hướng: {r['Goi_Y_Cho_SV']}.

Em vui lòng phản hồi email này và sắp xếp gặp Cố vấn học tập trong tuần này để cùng thống nhất kế hoạch đăng ký học phần phù hợp. Nếu em đang gặp khó khăn, em cứ chia sẻ để thầy/cô hỗ trợ kịp thời.

Trân trọng,
Cố vấn học tập - Học viện Ngân hàng."""


def page_intervention(df):
    st.title("🎯 Phân nhóm can thiệp & Phiếu tư vấn")
    g1 = df[df["Tin_Chi_No"] >= TH["debt_high"]]
    g2 = df[df["Ty_Le_Vang_Mat"] >= TH["absence_high"]]
    g3 = df[(df["GPA_Ky_Truoc"] < TH["gpa_low"]) & (df["Ty_Le_Vang_Mat"] < TH["diligent_absence"])]
    t1, t2, t3 = st.tabs([f"🚨 Nợ nhiều TC ({len(g1)} SV)",
                          f"📉 Chuyên cần kém ({len(g2)} SV)",
                          f"📚 Chăm học nhưng GPA thấp ({len(g3)} SV)"])
    with t1:
        st.write(f"**Tiêu chí:** nợ từ {TH['debt_high']} TC. **Hành động:** đề nghị giảm TC "
                 "đăng ký kỳ tới xuống 12-14 TC, ưu tiên học lại học phần tiên quyết.")
        show_df(g1[["Ma_SV", "Ho_Ten", "Lop_Nien_Che", "GPA_Ky_Truoc", "Tin_Chi_No",
                    "Tin_Chi_Dang_Ky", "De_Xuat_Cho_CVHT"]])
    with t2:
        st.write(f"**Tiêu chí:** vắng từ {TH['absence_high']}%. **Hành động:** nhắc nhở, tìm hiểu "
                 "nguyên nhân cá nhân (sức khỏe, đi làm thêm...) và thống nhất cam kết chuyên cần.")
        show_df(g2[["Ma_SV", "Ho_Ten", "Lop_Nien_Che", "Ty_Le_Vang_Mat", "GPA_Ky_Truoc",
                    "De_Xuat_Cho_CVHT"]])
    with t3:
        st.write(f"**Tiêu chí:** vắng < {TH['diligent_absence']}% nhưng GPA < {TH['gpa_low']}. "
                 "**Hành động:** hỗ trợ phương pháp học, ghép cặp với SV giỏi/CLB học thuật.")
        show_df(g3[["Ma_SV", "Ho_Ten", "Lop_Nien_Che", "GPA_Ky_Truoc", "Ty_Le_Vang_Mat",
                    "De_Xuat_Cho_CVHT"]])

    st.markdown("---")
    st.subheader("📝 Phiếu tư vấn & mẫu email cho từng sinh viên")
    d = df.sort_values("Diem_Rui_Ro", ascending=False)
    names = dict(zip(d["Ma_SV"], d["Ho_Ten"]))
    ma = st.selectbox("Chọn sinh viên:", d["Ma_SV"].tolist(),
                      format_func=lambda m: f"{m} - {names[m]}")
    r = d[d["Ma_SV"] == ma].iloc[0]
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔍 Hồ sơ & chẩn đoán")
        st.write(f"- **Họ tên:** {r['Ho_Ten']} ({r['Ma_SV']}) - Lớp {r['Lop_Nien_Che']}")
        st.write(f"- **Mức độ:** {r['Trang_Thai']} - điểm rủi ro {r['Diem_Rui_Ro']} "
                 f"({r['Nguon_Diem']})")
        st.write(f"- **Chỉ số:** GPA kỳ trước `{r['GPA_Ky_Truoc']}` | CPA `{r['GPA_Tich_Luy']}` | "
                 f"TC nợ `{int(r['Tin_Chi_No'])}` | Vắng `{int(r['Ty_Le_Vang_Mat'])}%`")
        st.write(f"- **Nguyên nhân chính:** {r['Nguyen_Nhan_Chinh']}")
        st.write(f"- **Khuyến nghị cho CVHT:** **{r['De_Xuat_Cho_CVHT']}**")
    with right:
        st.markdown("#### ✉️ Mẫu email (đã dùng ngôn ngữ trung tính, không nêu điểm rủi ro)")
        txt = st.text_area("Nội dung:", email_text(r), height=300, key=f"mail_{ma}")
        st.download_button("⬇️ Tải nội dung (.txt)", txt.encode("utf-8"), f"phieu_tu_van_{ma}.txt")


def whatif_panel(model, defaults, key, advisor_view):
    left, right = st.columns(2)
    with left:
        st.subheader("🎛️ Thông số giả định")
        gpa = st.slider("GPA kỳ trước (thang 4.0)", 0.0, 4.0, float(defaults["GPA_Ky_Truoc"]), 0.05, key=f"{key}_g")
        cpa = st.slider("GPA tích lũy (CPA)", 0.0, 4.0, float(defaults["GPA_Tich_Luy"]), 0.05, key=f"{key}_c")
        tcdk = st.slider("Số TC dự kiến đăng ký", 6, 27, int(np.clip(defaults["Tin_Chi_Dang_Ky"], 6, 27)), 1, key=f"{key}_d")
        tcno = st.slider("Số TC nợ", 0, 30, int(np.clip(defaults["Tin_Chi_No"], 0, 30)), 1, key=f"{key}_n")
        vang = st.slider("Tỷ lệ vắng dự kiến (%)", 0, 50, int(np.clip(defaults["Ty_Le_Vang_Mat"], 0, 50)), 1, key=f"{key}_v")
        drl = st.slider("Điểm rèn luyện", 30, 100, int(np.clip(defaults["Diem_Ren_Luyen"], 30, 100)), 1, key=f"{key}_r")
    with right:
        st.subheader("🤖 Kết quả mô phỏng")
        sim = pd.DataFrame([{"GPA_Ky_Truoc": gpa, "GPA_Tich_Luy": cpa, "Tin_Chi_Dang_Ky": tcdk,
                             "Tin_Chi_No": tcno, "Ty_Le_Vang_Mat": vang, "Diem_Ren_Luyen": drl}])
        res = analyze(sim, model).iloc[0]
        score, level = res["Diem_Rui_Ro"], res["Trang_Thai"]
        if advisor_view:
            st.metric("Điểm rủi ro cảnh báo học vụ", f"{score}")
            st.progress(float(np.clip(score / 100, 0, 1)))
        {RED: st.error, YELLOW: st.warning, GREEN: st.success}[level](
            STUDENT_LABEL[level] if not advisor_view else f"MỨC ĐỘ: {level}")
        st.write(f"**Chẩn đoán:** {res['Nguyen_Nhan_Chinh']}")
        st.write(f"**{'Khuyến nghị cho CVHT' if advisor_view else 'Gợi ý cho bạn'}:** "
                 f"{res['De_Xuat_Cho_CVHT'] if advisor_view else res['Goi_Y_Cho_SV']}")
        st.caption("Kết quả mang tính tham khảo để định hướng, không phải quyết định học vụ chính thức.")


def page_whatif(df, model):
    st.title("🔮 Mô phỏng kịch bản (What-if)")
    st.markdown("Điều chỉnh thử các thông số khi trao đổi với sinh viên để xem mức rủi ro thay đổi ra sao.")
    d = df.sort_values("Diem_Rui_Ro", ascending=False)
    names = dict(zip(d["Ma_SV"], d["Ho_Ten"]))
    opts = ["(Nhập thủ công)"] + d["Ma_SV"].tolist()
    ma = st.selectbox("Lấy số liệu khởi tạo từ sinh viên:", opts,
                      format_func=lambda m: m if m == opts[0] else f"{m} - {names[m]}")
    if ma == opts[0]:
        defaults = {"GPA_Ky_Truoc": 1.3, "GPA_Tich_Luy": 1.5, "Tin_Chi_Dang_Ky": 15,
                    "Tin_Chi_No": 9, "Ty_Le_Vang_Mat": 20, "Diem_Ren_Luyen": 65}
    else:
        defaults = d[d["Ma_SV"] == ma].iloc[0].to_dict()
    whatif_panel(model, defaults, key=f"wf_{ma}", advisor_view=True)


# ---------- Quản trị: nạp dữ liệu ----------
@st.cache_data(show_spinner=False)
def read_upload(name, data, sheet=None):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def page_data(user):
    st.title("📂 Nạp & chuẩn hóa dữ liệu")
    meta = load_meta()
    if dataset_mtime():
        st.info(f"Dữ liệu hiện hành: **{meta.get('source', '?')}** - {meta.get('rows', '?')} SV, "
                f"nạp lúc {meta.get('uploaded_at', '?')} bởi {meta.get('uploaded_by', '?')}. "
                "Sinh viên sẽ xem dữ liệu này khi đăng nhập.")
    else:
        st.warning("Chưa có dữ liệu. Hãy tải file lên hoặc nạp dữ liệu mô phỏng để thử.")

    sample = generate_sample_dataset()
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Tải file CSV mẫu", sample.to_csv(index=False).encode("utf-8-sig"),
                       "du_lieu_hoc_vu_mau.csv", "text/csv")
    if c2.button("🧪 Nạp dữ liệu mô phỏng (chỉ để thử)"):
        save_dataset(sample, "Dữ liệu MÔ PHỎNG (ngẫu nhiên)", user["username"])
        st.success("Đã nạp dữ liệu mô phỏng.")
        st.rerun()
    if dataset_mtime() and c3.button("🗑️ Xóa dữ liệu hiện hành"):
        DATASET_PATH.unlink(missing_ok=True)
        META_PATH.unlink(missing_ok=True)
        st.rerun()

    up = st.file_uploader("Chọn file dữ liệu lớp (.xlsx, .xls hoặc .csv):", type=["csv", "xlsx", "xls"])
    if up is None:
        return
    data = up.getvalue()
    sheet = None
    try:
        if up.name.lower().endswith((".xlsx", ".xls")):
            sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
            sheet = st.selectbox("Chọn sheet:", sheets) if len(sheets) > 1 else sheets[0]
        raw = read_upload(up.name, data, sheet)
    except Exception as e:  # noqa: BLE001
        st.error(f"Không đọc được file: {e}")
        return
    st.success(f"Đã đọc {len(raw)} dòng, {len(raw.columns)} cột.")
    show_df(raw.head(5))

    st.markdown("#### 🔗 Ánh xạ cột (đã tự đoán - hãy kiểm tra lại)")
    guess = auto_map(list(raw.columns))
    none_opt = "(không có)"
    mapping = {}
    cols = st.columns(2)
    for i, (std, (desc, req, _)) in enumerate(COLUMN_SPEC.items()):
        options = [none_opt] + [str(c) for c in raw.columns]
        idx = options.index(str(guess[std])) if guess[std] is not None else 0
        with cols[i % 2]:
            pick = st.selectbox(f"{desc}{' *' if req else ''}", options, index=idx,
                                key=f"map_{up.name}_{std}")
        mapping[std] = None if pick == none_opt else raw.columns[options.index(pick) - 1]

    if st.button("✅ Kiểm tra & lưu dữ liệu", type="primary"):
        df, errors, warns = standardize(raw, mapping)
        for e in errors:
            st.error(e)
        for w in warns:
            st.warning(w)
        if df is not None:
            save_dataset(df, up.name, user["username"])
            st.success(f"Đã lưu {len(df)} sinh viên. Vào mục 'Quản lý tài khoản' để cấp tài khoản cho SV.")
            st.balloons()


# ---------- Quản trị: tài khoản ----------
def page_accounts(user, df_all):
    st.title("👥 Quản lý tài khoản")
    t1, t2, t3 = st.tabs(["🎓 Cấp tài khoản sinh viên", "👨‍🏫 Thêm CVHT/Giảng viên", "📋 Danh sách & cấp lại mật khẩu"])

    with t1:
        if df_all is None:
            st.warning("Cần nạp dữ liệu trước.")
        else:
            st.write("Tạo tài khoản cho mọi SV trong dữ liệu chưa có tài khoản. Tên đăng nhập = "
                     "**Mã SV**, mật khẩu tạm ngẫu nhiên (SV bị buộc đổi ở lần đăng nhập đầu).")
            if st.button("⚙️ Tạo tài khoản hàng loạt"):
                creds = create_student_accounts(df_all)
                st.session_state["new_creds"] = creds
                if creds.empty:
                    st.info("Tất cả sinh viên đã có tài khoản.")
        creds = st.session_state.get("new_creds")
        if creds is not None and not creds.empty:
            st.warning("⚠️ Mật khẩu tạm chỉ hiển thị MỘT LẦN. Hãy tải về và gửi riêng cho từng SV.")
            show_df(creds)
            st.download_button("⬇️ Tải danh sách tài khoản (.CSV)",
                               creds.to_csv(index=False).encode("utf-8-sig"), "tai_khoan_sinh_vien.csv")
            if st.button("Đã lưu - ẩn danh sách"):
                st.session_state.pop("new_creds")
                st.rerun()

    with t2:
        classes_all = sorted(df_all["Lop_Nien_Che"].unique()) if df_all is not None else []
        with st.form("new_advisor"):
            u = st.text_input("Tên đăng nhập")
            name = st.text_input("Họ tên")
            role = st.selectbox("Vai trò", ["advisor (xem lớp được phân công)", "admin (toàn quyền)"])
            cls = st.multiselect("Lớp phụ trách (để trống = tất cả lớp)", classes_all)
            pw = st.text_input("Mật khẩu tạm", type="password")
            if st.form_submit_button("Tạo tài khoản"):
                if not u.strip():
                    st.error("Nhập tên đăng nhập.")
                elif get_user(u):
                    st.error("Tên đăng nhập đã tồn tại.")
                elif password_error(pw):
                    st.error(password_error(pw))
                else:
                    create_user(u, pw, role.split()[0], name, ",".join(cls), must_change=1)
                    st.success("Đã tạo. Người dùng sẽ phải đổi mật khẩu ở lần đăng nhập đầu.")

    with t3:
        users = pd.DataFrame(db_query(
            "SELECT username, role, full_name, classes, student_id, created_at FROM users ORDER BY role, username"))
        flt = st.selectbox("Lọc vai trò:", ["Tất cả", "admin", "advisor", "student"])
        view = users if flt == "Tất cả" else users[users["role"] == flt]
        show_df(view, height=300)
        target = st.selectbox("Chọn tài khoản:", users["username"].tolist())
        a, b = st.columns(2)
        if a.button("🔑 Cấp lại mật khẩu tạm"):
            pw = secrets.token_urlsafe(6)
            set_password(target, pw, must_change=1)
            st.success(f"Mật khẩu tạm của **{target}**: `{pw}` (chỉ hiển thị lần này)")
        if b.button("🗑️ Xóa tài khoản"):
            if target == user["username"]:
                st.error("Không thể tự xóa tài khoản đang đăng nhập.")
            elif get_user(target)["role"] == "admin" and \
                    db_query("SELECT COUNT(*) AS n FROM users WHERE role='admin'")[0]["n"] <= 1:
                st.error("Không thể xóa quản trị viên cuối cùng.")
            else:
                db_exec("DELETE FROM users WHERE username=?", (target,))
                st.success(f"Đã xóa {target}.")
                st.rerun()


# =========================================================
# 8. TRANG DÀNH CHO SINH VIÊN (chỉ xem hồ sơ của chính mình)
# =========================================================
def student_portal(user, df_all, model):
    st.title("🎓 Sức khỏe học vụ cá nhân")
    if df_all is None:
        st.info("Cố vấn học tập chưa cập nhật dữ liệu học kỳ này. Bạn quay lại sau nhé.")
        return
    mine = df_all[df_all["Ma_SV"].str.lower() == user["student_id"].lower()]
    if mine.empty:
        st.warning("Chưa tìm thấy dữ liệu của bạn trong kỳ này. Hãy liên hệ CVHT.")
        return
    sv = mine.iloc[0]
    st.markdown(f"Xin chào **{sv['Ho_Ten']}** - Lớp {sv['Lop_Nien_Che']} - Mã SV {sv['Ma_SV']}")
    st.info("Đây là công cụ tham khảo giúp bạn tự theo dõi tiến độ; không phải quyết định "
            "học vụ chính thức. Chỉ bạn và CVHT được xem thông tin này.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GPA kỳ trước", sv["GPA_Ky_Truoc"])
    c2.metric("GPA tích lũy", sv["GPA_Tich_Luy"])
    c3.metric("TC nợ", int(sv["Tin_Chi_No"]))
    c4.metric("Vắng mặt", f"{int(sv['Ty_Le_Vang_Mat'])}%")
    c5.metric("Trạng thái", STUDENT_LABEL[sv["Trang_Thai"]])

    cls = df_all[df_all["Lop_Nien_Che"] == sv["Lop_Nien_Che"]]
    left, right = st.columns(2)
    with left:
        st.subheader("📊 Bạn so với trung bình lớp")
        comp = pd.DataFrame({
            "Tiêu chí": ["GPA kỳ trước (quy đổi /100)", "Điểm rèn luyện", "Chuyên cần (%)"],
            "Bạn": [sv["GPA_Ky_Truoc"] / 4 * 100, sv["Diem_Ren_Luyen"], 100 - sv["Ty_Le_Vang_Mat"]],
            "Trung bình lớp": [cls["GPA_Ky_Truoc"].mean() / 4 * 100, cls["Diem_Ren_Luyen"].mean(),
                               100 - cls["Ty_Le_Vang_Mat"].mean()],
        }).melt(id_vars="Tiêu chí", var_name="Đối tượng", value_name="Giá trị")
        show_chart(px.bar(comp, x="Tiêu chí", y="Giá trị", color="Đối tượng", barmode="group"))
    with right:
        st.subheader("💡 Phân tích & gợi ý cho bạn")
        st.write(f"**Điểm cần lưu ý:** {sv['Nguyen_Nhan_Chinh']}")
        st.info(f"**Gợi ý hành động kỳ tới:** {sv['Goi_Y_Cho_SV']}")
        if sv["Trang_Thai"] != GREEN:
            st.write("👉 Hãy chủ động liên hệ Cố vấn học tập để được hỗ trợ - gặp sớm luôn dễ hơn.")

    st.markdown("---")
    st.subheader("🔮 Thử điều chỉnh kế hoạch của bạn")
    whatif_panel(model, sv.to_dict(), key="stu_wf", advisor_view=False)


# =========================================================
# 9. ĐIỀU PHỐI CHÍNH
# =========================================================
init_db()
if not db_query("SELECT 1 FROM users WHERE role='admin' LIMIT 1"):
    setup_screen()
    st.stop()

user = get_user(st.session_state.get("uname"))
if user is None:
    login_screen()
    st.stop()
if user["must_change"]:
    force_change_screen(user)
    st.stop()

# --- Tải dữ liệu & mô hình ---
mt = dataset_mtime()
model, model_info = get_model(mt) if mt else (None, None)
df_all = analyzed_dataset(mt, model) if mt else None
meta = load_meta()
source_text = f"`{meta.get('source', '?')}` (nạp {meta.get('uploaded_at', '?')})"

# --- Thanh bên ---
brand_header()
st.sidebar.markdown("---")
role_label = {"admin": "CVHT quản trị", "advisor": "CVHT/Giảng viên", "student": "Sinh viên"}[user["role"]]
st.sidebar.markdown(f"👤 **{user['full_name'] or user['username']}**  \n*{role_label}*")
with st.sidebar.expander("🔐 Đổi mật khẩu"):
    password_form(user)
if st.sidebar.button("Đăng xuất"):
    st.session_state.pop("uname", None)
    st.rerun()
st.sidebar.markdown("---")

# --- Sinh viên ---
if user["role"] == "student":
    student_portal(user, df_all, model)
    st.stop()

# --- CVHT / Giảng viên / Admin ---
pages = ["📊 Phân tích tổng quan", "🎯 Can thiệp & Tư vấn", "🔮 Mô phỏng kịch bản"]
if user["role"] == "admin":
    pages += ["📂 Nạp dữ liệu", "👥 Quản lý tài khoản"]
menu = st.sidebar.radio("📌 Chức năng", pages)

if menu == "📂 Nạp dữ liệu":
    page_data(user)
    st.stop()
if menu == "👥 Quản lý tài khoản":
    page_accounts(user, df_all)
    st.stop()

if df_all is None:
    st.warning("Chưa có dữ liệu. " + ("Vào mục 'Nạp dữ liệu' để tải file lên."
               if user["role"] == "admin" else "Hãy liên hệ quản trị viên để nạp dữ liệu."))
    st.stop()

df_scope = scope_df(df_all, user)
if df_scope.empty:
    st.warning("Bạn chưa được phân công lớp nào có trong dữ liệu hiện tại.")
    st.stop()

lop_chon = st.selectbox("🏫 Chọn lớp niên chế:", ["Tất cả các lớp"] + sorted(df_scope["Lop_Nien_Che"].unique()))
df_lop = df_scope if lop_chon == "Tất cả các lớp" else df_scope[df_scope["Lop_Nien_Che"] == lop_chon]
if df_lop.empty:
    st.warning("Lớp này không có sinh viên.")
    st.stop()

if menu == "📊 Phân tích tổng quan":
    page_overview(df_lop, lop_chon, model_info, source_text)
elif menu == "🎯 Can thiệp & Tư vấn":
    page_intervention(df_lop)
else:
    page_whatif(df_lop, model)

st.markdown("---")
st.caption("⚖️ Công cụ hỗ trợ ra quyết định: kết quả không thay thế đánh giá của CVHT. "
           "Dữ liệu sinh viên là thông tin nhạy cảm - không chia sẻ ngoài phạm vi công việc.")
