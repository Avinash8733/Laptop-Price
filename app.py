from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pickle
import streamlit as st


@dataclass(frozen=True)
class PredictionResult:
    price: float
    tier_label: str
    tier_color: str


def format_currency(amount: float, symbol: str = "₹") -> str:
    if not np.isfinite(amount):
        return f"{symbol}—"
    return f"{symbol}{amount:,.0f}"

def _prepare_dataset(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "").astype("int32")
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "").astype("float32")

    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in str(x) else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in str(x) else 0)

    new = df["ScreenResolution"].astype(str).str.split("x", n=1, expand=True)
    df["X_res"] = new[0]
    df["Y_res"] = new[1]
    df["X_res"] = df["X_res"].astype(str).str.replace(",", "").str.findall(r"(\d+\.?\d+)").apply(lambda x: x[0])
    df["X_res"] = df["X_res"].astype("int")
    df["Y_res"] = df["Y_res"].astype("int")

    df["ppi"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).astype("float")

    df.drop(columns=["ScreenResolution"], inplace=True)
    df.drop(columns=["Inches", "X_res", "Y_res"], inplace=True)

    df["Cpu Name"] = df["Cpu"].astype(str).apply(lambda x: " ".join(x.split()[0:3]))

    def fetch_processor(text: str) -> str:
        if text in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
            return text
        if text.split()[0] == "Intel":
            return "Other Intel Processor"
        return "AMD Processor"

    df["Cpu brand"] = df["Cpu Name"].apply(fetch_processor)
    df.drop(columns=["Cpu Name", "Cpu"], inplace=True)

    df["Memory"] = (
        df["Memory"]
        .astype(str)
        .str.replace(r"\.0", "", regex=True)
        .str.replace("GB", "", regex=False)
        .str.replace("TB", "000", regex=False)
    )

    new = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = new[0].str.strip()
    df["second"] = new[1].fillna("0")

    df["Layer1HDD"] = df["first"].str.contains("HDD", na=False).astype(int)
    df["Layer1SSD"] = df["first"].str.contains("SSD", na=False).astype(int)
    df["Layer1Hybrid"] = df["first"].str.contains("Hybrid", na=False).astype(int)
    df["Layer1Flash_Storage"] = df["first"].str.contains("Flash Storage", na=False).astype(int)

    df["Layer2HDD"] = df["second"].str.contains("HDD", na=False).astype(int)
    df["Layer2SSD"] = df["second"].str.contains("SSD", na=False).astype(int)
    df["Layer2Hybrid"] = df["second"].str.contains("Hybrid", na=False).astype(int)
    df["Layer2Flash_Storage"] = df["second"].str.contains("Flash Storage", na=False).astype(int)

    df["first"] = pd.to_numeric(df["first"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0)
    df["second"] = pd.to_numeric(df["second"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0)

    df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
    df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
    df["Hybrid"] = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
    df["Flash_Storage"] = df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]

    df.drop(
        columns=[
            "first",
            "second",
            "Layer1HDD",
            "Layer1SSD",
            "Layer1Hybrid",
            "Layer1Flash_Storage",
            "Layer2HDD",
            "Layer2SSD",
            "Layer2Hybrid",
            "Layer2Flash_Storage",
        ],
        inplace=True,
    )

    df.drop(columns=["Memory"], inplace=True)
    df.drop(columns=["Hybrid", "Flash_Storage"], inplace=True)

    df["Gpu brand"] = df["Gpu"].astype(str).apply(lambda x: x.split()[0])
    df = df[df["Gpu brand"] != "ARM"]
    df.drop(columns=["Gpu"], inplace=True)

    def cat_os(inp: str) -> str:
        if inp in ("Windows 10", "Windows 7", "Windows 10 S"):
            return "Windows"
        if inp in ("macOS", "Mac OS X"):
            return "Mac"
        return "Others/No OS/Linux"

    df["os"] = df["OpSys"].astype(str).apply(cat_os)
    df.drop(columns=["OpSys"], inplace=True)

    return df


def rebuild_artifacts(here: Path) -> tuple[object, object]:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = _prepare_dataset(here / "laptop_data.csv")

    X = df.drop(columns=["Price"])
    y = np.log(df["Price"])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=2)

    categorical_cols = ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"]
    step1 = ColumnTransformer(
        transformers=[
            ("col_tnf", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough",
    )

    step2 = RandomForestRegressor(
        n_estimators=350,
        random_state=3,
        max_samples=0.5,
        max_features=0.75,
        max_depth=15,
        bootstrap=True,
        n_jobs=-1,
    )

    pipe = Pipeline([("step1", step1), ("step2", step2)])
    pipe.fit(X_train, y_train)

    with (here / "df.pkl").open("wb") as f:
        pickle.dump(df, f)
    with (here / "pipe.pkl").open("wb") as f:
        pickle.dump(pipe, f)

    return pipe, df


def load_artifacts() -> tuple[object, object]:
    here = Path(__file__).resolve().parent
    try:
        with (here / "pipe.pkl").open("rb") as f:
            pipe = pickle.load(f)
        with (here / "df.pkl").open("rb") as f:
            df = pickle.load(f)
        return pipe, df
    except Exception:
        return rebuild_artifacts(here)


def get_model_name(pipe: object) -> str:
    step2 = getattr(pipe, "named_steps", {}).get("step2")
    if step2 is None:
        step2 = getattr(pipe, "steps", [("model", None)])[-1][1]
    return step2.__class__.__name__ if step2 is not None else "Model"


def price_tier(price: float, df_prices: np.ndarray) -> tuple[str, str]:
    if not np.isfinite(price) or df_prices.size == 0:
        return "Unknown", "#64748b"
    q1, q2 = np.quantile(df_prices, [0.33, 0.66])
    if price < q1:
        return "Budget", "#16a34a"
    if price < q2:
        return "Mid‑range", "#f59e0b"
    return "Premium", "#ef4444"


def predict_price(
    pipe: object,
    *,
    company: str,
    type_name: str,
    ram: int,
    weight: float,
    touchscreen: int,
    ips: int,
    screen_size: float,
    resolution: str,
    cpu: str,
    hdd: int,
    ssd: int,
    gpu: str,
    os_name: str,
    df_prices: np.ndarray,
) -> PredictionResult:
    import pandas as pd

    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])
    ppi = (((x_res**2) + (y_res**2)) ** 0.5) / float(screen_size)

    query = pd.DataFrame(
        [
            {
                "Company": company,
                "TypeName": type_name,
                "Ram": int(ram),
                "Weight": float(weight),
                "Touchscreen": int(touchscreen),
                "Ips": int(ips),
                "ppi": float(ppi),
                "Cpu brand": cpu,
                "HDD": int(hdd),
                "SSD": int(ssd),
                "Gpu brand": gpu,
                "os": os_name,
            }
        ]
    )

    predicted_log = float(pipe.predict(query)[0])
    predicted_price = float(np.exp(predicted_log))

    label, color = price_tier(predicted_price, df_prices)
    return PredictionResult(price=predicted_price, tier_label=label, tier_color=color)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1150px; }
          .app-header { text-align: center; margin-bottom: 1rem; }
          .app-title { font-size: 2.1rem; font-weight: 800; line-height: 1.1; margin: 0; }
          .app-subtitle { font-size: 1rem; opacity: 0.85; margin-top: 0.35rem; }

          div[data-testid="stForm"] { border: 1px solid rgba(148,163,184,.35); border-radius: 16px; padding: 1.1rem; background: rgba(2,6,23,.02); }
          div[data-testid="stMetric"] { border: 1px solid rgba(148,163,184,.35); border-radius: 16px; padding: 0.75rem 0.85rem; background: rgba(2,6,23,.02); }

          div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            font-weight: 700;
          }

          .tier-pill {
            display: inline-block;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            color: white;
            margin-left: 0.5rem;
            vertical-align: middle;
          }

          .hint { opacity: 0.85; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

here = Path(__file__).resolve().parent

with st.sidebar:
    st.markdown("### 🛠️ Maintenance")
    rebuild_clicked = st.button("🔄 Rebuild model files", use_container_width=True)
    if rebuild_clicked:
        with st.spinner("Rebuilding model artifacts..."):
            rebuild_artifacts(here)
        st.success("Model files rebuilt. Reloading...")
        st.rerun()

pipe, df = load_artifacts()
model_name = get_model_name(pipe)

df_prices = df["Price"].to_numpy(dtype=float) if "Price" in df.columns else np.array([])

st.markdown(
    """
    <div class="app-header">
      <div class="app-title">💻 Laptop Price Prediction</div>
      <div class="app-subtitle">A clean, responsive estimator powered by a trained ML pipeline.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📌 Project Overview")
    st.write("Predict a laptop’s expected price from key specifications using a trained machine learning pipeline.")

    st.markdown("### 🧭 Instructions")
    st.write("1) Choose specifications  2) Click **Predict Price**  3) Review the estimated price card.")

    st.markdown("### 🧠 Model Details")
    st.write(f"- Pipeline: Preprocessing + Regressor")
    st.write(f"- Regressor: **{model_name}**")

    st.markdown("### 👨‍💻 Developer")
    st.write("- Name: **parma**")
    st.write("- App: Streamlit")

    with st.expander("ℹ️ Environment", expanded=False):
        try:
            import sys
            import numpy as _np
            import sklearn as _sk

            st.caption(f"Python: {sys.version.split()[0]}")
            st.caption(f"Streamlit: {st.__version__}")
            st.caption(f"NumPy: {_np.__version__}")
            st.caption(f"scikit‑learn: {_sk.__version__}")
        except Exception:
            st.caption("Version info unavailable.")


col_left, col_right = st.columns([1.25, 0.75], gap="large")

with col_left:
    st.markdown("#### 🧩 Laptop Specifications")
    st.markdown('<div class="hint">Neatly fill the fields below. Advanced display options are optional.</div>', unsafe_allow_html=True)

    default_weight = float(df["Weight"].median()) if "Weight" in df.columns else 2.0
    ram_choices = sorted({int(x) for x in df["Ram"].dropna().unique()}) if "Ram" in df.columns else [8, 16, 32]
    hdd_choices = sorted({int(x) for x in df["HDD"].dropna().unique()}) if "HDD" in df.columns else [0, 512, 1024]
    ssd_choices = sorted({int(x) for x in df["SSD"].dropna().unique()}) if "SSD" in df.columns else [0, 256, 512, 1024]

    resolutions = [
        "1920x1080",
        "1366x768",
        "1600x900",
        "3840x2160",
        "3200x1800",
        "2880x1800",
        "2560x1600",
        "2560x1440",
        "2304x1440",
    ]

    with st.form("predict_form", clear_on_submit=False):
        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
        with r1c1:
            company = st.selectbox("🏷️ Brand", sorted(df["Company"].dropna().unique()))
        with r1c2:
            type_name = st.selectbox("🧰 Type", sorted(df["TypeName"].dropna().unique()))
        with r1c3:
            os_name = st.selectbox("🖥️ Operating System", sorted(df["os"].dropna().unique()))

        r2c1, r2c2 = st.columns(2, gap="medium")
        with r2c1:
            cpu = st.selectbox("🧠 Processor", sorted(df["Cpu brand"].dropna().unique()))
        with r2c2:
            gpu = st.selectbox("🎮 GPU", sorted(df["Gpu brand"].dropna().unique()))

        r3c1, r3c2, r3c3 = st.columns(3, gap="medium")
        with r3c1:
            ram = st.selectbox("🧷 RAM (GB)", ram_choices, index=ram_choices.index(8) if 8 in ram_choices else 0)
        with r3c2:
            weight = st.number_input("⚖️ Weight (kg)", min_value=0.5, max_value=6.0, value=float(default_weight), step=0.1)
        with r3c3:
            screen_size = st.slider("📏 Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)

        r4c1, r4c2 = st.columns(2, gap="medium")
        with r4c1:
            ssd = st.selectbox("⚡ SSD (GB)", ssd_choices, index=ssd_choices.index(256) if 256 in ssd_choices else 0)
        with r4c2:
            hdd = st.selectbox("💾 HDD (GB)", hdd_choices, index=hdd_choices.index(0) if 0 in hdd_choices else 0)

        with st.expander("🧪 Display & Extras (optional)", expanded=False):
            a1, a2, a3 = st.columns(3, gap="medium")
            with a1:
                touchscreen_ui = st.selectbox("👆 Touchscreen", ["No", "Yes"], index=0)
            with a2:
                ips_ui = st.selectbox("🪟 IPS Panel", ["No", "Yes"], index=0)
            with a3:
                resolution = st.selectbox("🧾 Resolution", resolutions, index=0)

        st.markdown("")
        try:
            predict_clicked = st.form_submit_button("🚀 Predict Price", type="primary", use_container_width=True)
        except TypeError:
            predict_clicked = st.form_submit_button("🚀 Predict Price", use_container_width=True)

with col_right:
    st.markdown("#### 💰 Predicted Price")
    st.markdown('<div class="hint">Your estimate appears here after prediction.</div>', unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if predict_clicked:
        touchscreen = 1 if touchscreen_ui == "Yes" else 0
        ips = 1 if ips_ui == "Yes" else 0
        try:
            st.session_state.last_result = predict_price(
                pipe,
                company=company,
                type_name=type_name,
                ram=int(ram),
                weight=float(weight),
                touchscreen=int(touchscreen),
                ips=int(ips),
                screen_size=float(screen_size),
                resolution=resolution,
                cpu=cpu,
                hdd=int(hdd),
                ssd=int(ssd),
                gpu=gpu,
                os_name=os_name,
                df_prices=df_prices,
            )
        except Exception as e:
            st.session_state.last_result = None
            st.exception(e)

    result: PredictionResult | None = st.session_state.last_result
    if result is None:
        st.metric("Estimated Price", "—")
        st.info("Tip: Fill specs and click **Predict Price**.", icon="💡")
    else:
        st.metric("Estimated Price", format_currency(result.price, "₹"))
        st.markdown(
            f'<div class="tier-pill" style="background:{result.tier_color};">{result.tier_label}</div>',
            unsafe_allow_html=True,
        )
        st.success("Prediction ready. Adjust specs to compare configurations.", icon="✅")

