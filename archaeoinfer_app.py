
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import json
import math

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="ArchaeoInfer | 考古推理平台",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Language helpers
# =========================
I18N = {
    "zh": {
        "title": "ArchaeoInfer：偏差校正考古推理平台",
        "subtitle": "一个中英文交互式 Streamlit 原型，用于遗址存在概率、可见性、结构补全、埋深估计与用途推断。",
        "disclaimer_title": "重要说明",
        "disclaimer_body": (
            "这是一个可运行的研究原型，而不是已经过考古实地验证的生产系统。"
            "其中“检索增强”和“图像用途推断”模块当前使用的是本地知识库 / 用户上传资料 / 规则推理的实现方式，"
            "没有直接接入开放互联网自动抓取。这样做是为了保持结果可解释、可追溯、便于后续替换为正式 API。"
        ),
        "sidebar_lang": "语言 / Language",
        "sidebar_upload": "上传数据",
        "dataset_help": "支持 CSV。若没有数据，可使用内置示例数据。",
        "use_demo": "使用内置示例数据",
        "uploaded_dataset": "已上传数据集预览",
        "demo_dataset": "当前使用：内置示例数据",
        "missing_cols_warning": "数据中缺少部分推荐字段。系统将对缺失字段自动补默认值。",
        "tab1": "1. 遗址存在概率",
        "tab2": "2. 可见性 / 可发现性",
        "tab3": "3. 局部结构补全",
        "tab4": "4. 埋深与年代估计",
        "tab5": "5. 图像 + 贝叶斯用途推断",
        "tab6": "6. 数据与结果下载",
        "region_model_title": "模块 1：遗址存在概率模型",
        "region_model_desc": "目标：估计古人为什么更可能在某处建立遗址。",
        "env_features": "环境约束层特征",
        "culture_features": "文化网络层特征",
        "time_feature": "时间层权重",
        "run_model": "运行模型",
        "exists_prob_done": "遗址存在概率已计算。",
        "detect_model_title": "模块 2：可见性 / 可发现性模型",
        "detect_model_desc": "目标：区分“真实存在”与“今天更容易被看到”。",
        "detect_prob_done": "可见性概率已计算。",
        "combined_prob_done": "观测概率已计算：P(observed)=P(exists)×P(detectable)",
        "structure_title": "模块 3：局部结构补全",
        "structure_desc": "根据已发现单元、对称性与文化模板，推断潜在缺失单元。",
        "burial_title": "模块 4：埋深与年代区间估计",
        "burial_desc": "联合估计埋深区间与年代区间。这里采用可解释的启发式/回归式原型。",
        "image_title": "模块 5：图像 + 贝叶斯用途推断",
        "image_desc": "上传遗迹/器物照片，并结合上下文、局部知识库和贝叶斯更新给出候选用途。",
        "download_title": "模块 6：下载结果",
        "download_desc": "下载当前结果表。",
        "select_features": "选择特征",
        "target_site": "遗址标签列（1=已知遗址，0=背景点）",
        "fit_success": "模型训练完成",
        "prob_col": "概率列",
        "top_zones": "高潜力区域（Top 10）",
        "show_map_note": "这里使用表格与散点近似表示热区；如需正式地图，请接入 GeoPandas/Folium。",
        "detect_features": "可见性特征",
        "combined_formula": "观测概率公式",
        "structure_settings": "结构补全参数",
        "coord_x": "X 坐标列",
        "coord_y": "Y 坐标列",
        "unit_type": "单元类型列",
        "culture_type": "文化类型",
        "symmetry_strength": "对称性权重",
        "neighbor_radius": "邻域半径",
        "suggested_units": "建议补全单元",
        "depth_age_output": "埋深与年代估计结果",
        "upload_image": "上传图片",
        "context_input": "上下文输入",
        "context_placeholder": "例如：出土于墓室北侧，附近有陶片和动物骨骼，地区为汉代聚落边缘。",
        "kb_upload": "上传本地知识库（CSV，可选）",
        "bayes_run": "运行用途推断",
        "usage_hypotheses": "候选用途概率",
        "evidence_chain": "证据链说明",
        "download_csv": "下载 CSV",
        "download_json": "下载 JSON",
        "image_features": "图像特征",
        "notes": "备注",
        "default_kb": "使用内置简化知识库",
        "not_enough_cols": "至少需要 x, y 两列坐标用于结构补全。",
        "no_image_uploaded": "尚未上传图片。图像特征将留空，仅根据上下文与知识库推断。",
    },
    "en": {
        "title": "ArchaeoInfer: Bias-Aware Archaeological Inference Platform",
        "subtitle": "A bilingual Streamlit prototype for site existence probability, detectability, structural completion, burial-depth estimation, and function inference.",
        "disclaimer_title": "Important note",
        "disclaimer_body": (
            "This is a research prototype, not a field-validated production system. "
            "The retrieval-augmented and image-based inference modules currently rely on a local knowledge base / user uploads / rule-based reasoning, "
            "rather than unrestricted internet crawling. This keeps the workflow interpretable, traceable, and easy to upgrade later."
        ),
        "sidebar_lang": "Language / 语言",
        "sidebar_upload": "Upload data",
        "dataset_help": "CSV supported. You may also use the built-in demo dataset.",
        "use_demo": "Use built-in demo dataset",
        "uploaded_dataset": "Uploaded dataset preview",
        "demo_dataset": "Current dataset: built-in demo",
        "missing_cols_warning": "Some recommended columns are missing. Default values will be injected automatically.",
        "tab1": "1. Site Existence",
        "tab2": "2. Detectability",
        "tab3": "3. Structural Completion",
        "tab4": "4. Burial Depth & Age",
        "tab5": "5. Image + Bayesian Function Inference",
        "tab6": "6. Download",
        "region_model_title": "Module 1: Site Existence Probability",
        "region_model_desc": "Goal: estimate why ancient people were more likely to choose certain locations.",
        "env_features": "Environmental constraint features",
        "culture_features": "Cultural network features",
        "time_feature": "Temporal-weight feature",
        "run_model": "Run model",
        "exists_prob_done": "Site existence probability computed.",
        "detect_model_title": "Module 2: Detectability",
        "detect_model_desc": "Goal: separate true presence from present-day visibility/discoverability.",
        "detect_prob_done": "Detectability probability computed.",
        "combined_prob_done": "Observed probability computed: P(observed)=P(exists)×P(detectable)",
        "structure_title": "Module 3: Local Structural Completion",
        "structure_desc": "Infer likely missing units from discovered units, symmetry, and cultural templates.",
        "burial_title": "Module 4: Burial Depth & Age Estimation",
        "burial_desc": "Jointly estimate burial-depth range and age range using an interpretable heuristic/regression prototype.",
        "image_title": "Module 5: Image + Bayesian Function Inference",
        "image_desc": "Upload an artifact/remain image, combine context and a local knowledge base, then infer candidate functions.",
        "download_title": "Module 6: Download Results",
        "download_desc": "Download current outputs.",
        "select_features": "Select features",
        "target_site": "Site label column (1=known site, 0=background)",
        "fit_success": "Model fit complete",
        "prob_col": "Probability column",
        "top_zones": "Top 10 high-potential zones",
        "show_map_note": "This prototype uses tables/scatter approximation; for production maps, connect GeoPandas/Folium.",
        "detect_features": "Detectability features",
        "combined_formula": "Observed probability formula",
        "structure_settings": "Structural completion settings",
        "coord_x": "X coordinate column",
        "coord_y": "Y coordinate column",
        "unit_type": "Unit-type column",
        "culture_type": "Culture type",
        "symmetry_strength": "Symmetry weight",
        "neighbor_radius": "Neighborhood radius",
        "suggested_units": "Suggested missing units",
        "depth_age_output": "Burial-depth & age estimation",
        "upload_image": "Upload image",
        "context_input": "Context input",
        "context_placeholder": "Example: found on the north side of a tomb chamber, near pottery sherds and animal bones, on the edge of a Han settlement.",
        "kb_upload": "Upload local knowledge base (CSV, optional)",
        "bayes_run": "Run function inference",
        "usage_hypotheses": "Function hypothesis probabilities",
        "evidence_chain": "Evidence chain",
        "download_csv": "Download CSV",
        "download_json": "Download JSON",
        "image_features": "Image features",
        "notes": "Notes",
        "default_kb": "Using built-in simplified knowledge base",
        "not_enough_cols": "At least x and y coordinate columns are required for structural completion.",
        "no_image_uploaded": "No image uploaded. Image features will remain empty and inference will rely on context + knowledge base.",
    },
}


def tr(key: str) -> str:
    return I18N[st.session_state.lang][key]


if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# =========================
# Utility functions
# =========================
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(series.median() if pd.api.types.is_numeric_dtype(series) else 0)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(series.median() if pd.api.types.is_numeric_dtype(series) else 0)
    std = s.std(ddof=0)
    if std < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().all():
            out[c] = 0.0
        else:
            out[c] = out[c].fillna(out[c].median())
    return out


def create_demo_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)

    dist_to_river = np.abs(y - 50) + rng.normal(0, 3, n)
    slope = np.clip(rng.normal(12, 5, n), 0, 30)
    soil_quality = np.clip(rng.normal(0.55, 0.18, n), 0, 1)
    elevation = np.clip(40 + 0.6 * x + rng.normal(0, 8, n), 0, 120)
    dist_to_ancient_route = np.abs(x - 35) + rng.normal(0, 4, n)
    dist_to_political_center = np.sqrt((x - 25) ** 2 + (y - 70) ** 2)
    dist_to_ritual_landscape = np.sqrt((x - 75) ** 2 + (y - 25) ** 2)
    resource_access = np.clip(1 - np.sqrt((x - 60) ** 2 + (y - 60) ** 2) / 100 + rng.normal(0, 0.05, n), 0, 1)
    time_weight = rng.choice([0.2, 0.5, 0.8, 1.0], size=n, p=[0.15, 0.35, 0.3, 0.2])

    # Detectability variables
    ndvi = np.clip(rng.normal(0.55, 0.2, n), 0, 1)
    soil_moisture = np.clip(rng.normal(0.45, 0.18, n), 0, 1)
    modern_coverage = np.clip(rng.normal(0.30, 0.2, n), 0, 1)
    erosion_risk = np.clip(rng.normal(0.35, 0.18, n), 0, 1)
    season_visibility = np.clip(1 - np.abs(ndvi - 0.35), 0, 1)

    # Unit layout fields
    unit_type = rng.choice(["room", "corridor", "chamber", "pit"], size=n, p=[0.3, 0.2, 0.25, 0.25])
    culture_type = rng.choice(["Han", "Tang", "RitualComplex"], size=n, p=[0.45, 0.35, 0.20])

    # Existence latent score
    z_exists = (
        -0.04 * dist_to_river
        -0.05 * slope
        +1.7 * soil_quality
        -0.03 * dist_to_ancient_route
        -0.025 * dist_to_political_center
        -0.01 * dist_to_ritual_landscape
        +1.4 * resource_access
        +1.0 * time_weight
        -0.005 * np.abs(elevation - 65)
    )
    p_exists = sigmoid(z_exists)
    is_site = (rng.random(n) < p_exists * 0.75).astype(int)

    # Detectability latent score
    z_detect = (
        -1.2 * ndvi
        +1.0 * season_visibility
        -1.3 * modern_coverage
        -0.7 * erosion_risk
        +0.6 * soil_moisture
        +0.25 * (1 - np.clip(dist_to_river / 100, 0, 1))
    )
    p_detect = sigmoid(z_detect)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "dist_to_river": dist_to_river,
        "slope": slope,
        "soil_quality": soil_quality,
        "elevation": elevation,
        "dist_to_ancient_route": dist_to_ancient_route,
        "dist_to_political_center": dist_to_political_center,
        "dist_to_ritual_landscape": dist_to_ritual_landscape,
        "resource_access": resource_access,
        "time_weight": time_weight,
        "ndvi": ndvi,
        "soil_moisture": soil_moisture,
        "modern_coverage": modern_coverage,
        "erosion_risk": erosion_risk,
        "season_visibility": season_visibility,
        "unit_type": unit_type,
        "culture_type": culture_type,
        "is_site": is_site,
    })
    return df


def ensure_recommended_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "x": np.arange(len(df)),
        "y": np.zeros(len(df)),
        "dist_to_river": 50.0,
        "slope": 10.0,
        "soil_quality": 0.5,
        "elevation": 60.0,
        "dist_to_ancient_route": 50.0,
        "dist_to_political_center": 60.0,
        "dist_to_ritual_landscape": 60.0,
        "resource_access": 0.5,
        "time_weight": 0.5,
        "ndvi": 0.5,
        "soil_moisture": 0.4,
        "modern_coverage": 0.3,
        "erosion_risk": 0.3,
        "season_visibility": 0.5,
        "unit_type": "room",
        "culture_type": "Generic",
        "is_site": 0,
    }
    out = df.copy()
    missing = []
    for c, v in defaults.items():
        if c not in out.columns:
            out[c] = v
            missing.append(c)
    return out, missing


@dataclass
class WeightedPrototypeModel:
    weights: Dict[str, float]
    inverse_features: Tuple[str, ...] = ()

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        df_num = safe_numeric(df, list(self.weights.keys()))
        score = np.zeros(len(df_num))
        total_abs = sum(abs(v) for v in self.weights.values()) or 1.0
        for feat, w in self.weights.items():
            scaled = minmax(df_num[feat]).values
            if feat in self.inverse_features:
                scaled = 1 - scaled
            score += w * scaled
        score = score / total_abs
        return pd.Series(sigmoid(4 * (score - 0.5)), index=df.index)


def fit_simple_logistic(df: pd.DataFrame, features: List[str], target: str, lr: float = 0.1, epochs: int = 1000) -> Tuple[np.ndarray, float]:
    X = safe_numeric(df, features)[features].copy()
    X = np.column_stack([np.ones(len(X)), *(zscore(X[c]).values for c in features)])
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(float).values
    w = np.zeros(X.shape[1])

    for _ in range(epochs):
        preds = sigmoid(X @ w)
        grad = X.T @ (preds - y) / len(y)
        w -= lr * grad

    intercept = w[0]
    coefs = w[1:]
    return coefs, intercept


def predict_simple_logistic(df: pd.DataFrame, features: List[str], coefs: np.ndarray, intercept: float) -> pd.Series:
    X = safe_numeric(df, features)[features].copy()
    X = np.column_stack([*(zscore(X[c]).values for c in features)])
    z = intercept + X @ coefs
    return pd.Series(sigmoid(z), index=df.index)


def compute_exists_probability(df: pd.DataFrame, env_features: List[str], culture_features: List[str], time_feature: str, target: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    features = env_features + culture_features + [time_feature]
    coefs, intercept = fit_simple_logistic(df, features, target)
    probs = predict_simple_logistic(df, features, coefs, intercept)
    out = df.copy()
    out["p_exists"] = probs
    coef_map = {f: float(c) for f, c in zip(features, coefs)}
    coef_map["intercept"] = float(intercept)
    return out, coef_map


def compute_detectability(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    default_weights = {}
    for f in features:
        if "coverage" in f or "ndvi" in f or "erosion" in f:
            default_weights[f] = -1.0
        else:
            default_weights[f] = 1.0
    inverse = tuple([f for f in features if any(k in f.lower() for k in ["dist_", "coverage", "ndvi", "erosion"])])
    model = WeightedPrototypeModel(weights=default_weights, inverse_features=inverse)
    out = df.copy()
    out["p_detectable"] = model.predict_proba(out[features])
    return out, default_weights


def combine_observed_probability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_observed"] = out["p_exists"] * out["p_detectable"]
    return out


def structural_completion(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    unit_col: str,
    culture_value: str,
    symmetry_strength: float,
    radius: float,
) -> pd.DataFrame:
    pts = df[[x_col, y_col, unit_col, "culture_type"]].copy()
    pts = pts[pts["culture_type"].astype(str) == str(culture_value)].copy()
    if len(pts) < 2:
        return pd.DataFrame(columns=["suggested_x", "suggested_y", "score", "reason", "suggested_unit_type"])

    center_x = pts[x_col].mean()
    center_y = pts[y_col].mean()

    suggestions = []
    arr = pts[[x_col, y_col]].values
    for _, row in pts.iterrows():
        sx = 2 * center_x - row[x_col]
        sy = 2 * center_y - row[y_col]
        dists = np.sqrt(((arr[:, 0] - sx) ** 2) + ((arr[:, 1] - sy) ** 2))
        nearest = dists.min() if len(dists) else np.inf
        if nearest > radius * 0.6:
            local_neighbors = np.sqrt(((arr[:, 0] - row[x_col]) ** 2) + ((arr[:, 1] - row[y_col]) ** 2))
            density_score = float(np.sum(local_neighbors < radius)) / max(len(arr), 1)
            score = symmetry_strength * (1 / (1 + nearest)) + (1 - symmetry_strength) * density_score
            suggestions.append({
                "suggested_x": round(float(sx), 3),
                "suggested_y": round(float(sy), 3),
                "score": round(float(score), 4),
                "reason": "symmetry + local density",
                "suggested_unit_type": row[unit_col],
            })

    if not suggestions:
        return pd.DataFrame(columns=["suggested_x", "suggested_y", "score", "reason", "suggested_unit_type"])

    sug_df = pd.DataFrame(suggestions)
    sug_df = sug_df.sort_values("score", ascending=False).drop_duplicates(subset=["suggested_x", "suggested_y"]).head(20)
    return sug_df


def estimate_depth_and_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = ["soil_moisture", "erosion_risk", "elevation", "dist_to_river", "modern_coverage", "slope"]
    out = safe_numeric(out, [c for c in numeric_cols if c in out.columns])

    # Heuristic interpretable estimates
    depth_base = (
        0.8
        + 2.0 * minmax(out["soil_moisture"])
        + 1.2 * (1 - minmax(out["erosion_risk"]))
        + 0.6 * (1 - minmax(out["modern_coverage"]))
        + 0.5 * (1 - minmax(out["slope"]))
        + 0.7 * (1 - minmax(out["dist_to_river"]))
    )
    out["depth_est_m"] = depth_base.round(2)
    out["depth_low_m"] = (out["depth_est_m"] - 0.5).clip(lower=0)
    out["depth_high_m"] = (out["depth_est_m"] + 0.7)

    time_weight = pd.to_numeric(out["time_weight"], errors="coerce").fillna(0.5)
    age_center = 500 + 1400 * time_weight + 250 * (1 - minmax(out["modern_coverage"]))
    out["estimated_age_bp"] = age_center.round(0).astype(int)
    out["age_low_bp"] = (out["estimated_age_bp"] - 180).clip(lower=0)
    out["age_high_bp"] = out["estimated_age_bp"] + 220
    return out


def extract_basic_image_features(image_bytes: bytes) -> Dict[str, float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image.resize((256, 256))).astype(np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))
    gray = arr.mean(axis=2)
    contrast = float(gray.std())
    edge_proxy = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    symmetry_h = float(1 - np.abs(gray - np.fliplr(gray)).mean())
    symmetry_v = float(1 - np.abs(gray - np.flipud(gray)).mean())
    dark_ratio = float((gray < 0.35).mean())
    light_ratio = float((gray > 0.7).mean())

    return {
        "mean_r": round(float(mean_rgb[0]), 4),
        "mean_g": round(float(mean_rgb[1]), 4),
        "mean_b": round(float(mean_rgb[2]), 4),
        "contrast": round(contrast, 4),
        "edge_proxy": round(edge_proxy, 4),
        "symmetry_h": round(symmetry_h, 4),
        "symmetry_v": round(symmetry_v, 4),
        "dark_ratio": round(dark_ratio, 4),
        "light_ratio": round(light_ratio, 4),
    }


def default_knowledge_base() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "hypothesis": "ritual",
            "keywords": "bones altar offering incense symmetric ritual ceremonial sacrifice painted",
            "context_cues": "north chamber ritual landscape offering animal bones",
            "feature_rule": "high_symmetry",
            "prior": 0.22,
            "note_zh": "若有祭祀景观、动物骨骼、较强对称性，则祭祀概率上升。",
            "note_en": "Ritual probability rises with ritual landscapes, animal bones, and stronger symmetry.",
        },
        {
            "hypothesis": "storage",
            "keywords": "jar container granary storage pit utilitarian clustered pottery",
            "context_cues": "storage pit edge settlement jars",
            "feature_rule": "medium_symmetry",
            "prior": 0.18,
            "note_zh": "若附近有储藏坑、容器类器物，则储藏用途上升。",
            "note_en": "Storage probability rises near pits, jars, and utilitarian containers.",
        },
        {
            "hypothesis": "residential",
            "keywords": "hearth domestic room dwelling ash floor corridor",
            "context_cues": "settlement room domestic floor hearth",
            "feature_rule": "low_to_medium_symmetry",
            "prior": 0.22,
            "note_zh": "若有灶、地面遗迹、居住区线索，则居住用途上升。",
            "note_en": "Residential probability rises with hearths, floors, and settlement context.",
        },
        {
            "hypothesis": "burial_associated",
            "keywords": "tomb chamber coffin burial grave goods north side south side",
            "context_cues": "tomb chamber grave goods burial tomb",
            "feature_rule": "high_symmetry",
            "prior": 0.23,
            "note_zh": "墓室、随葬品、方位性强时，墓葬附属用途更可能。",
            "note_en": "Burial-associated probability rises in tomb contexts with grave goods and strong orientation.",
        },
        {
            "hypothesis": "production",
            "keywords": "kiln slag workshop manufacture furnace tool",
            "context_cues": "slag ash workshop kiln furnace",
            "feature_rule": "low_symmetry",
            "prior": 0.15,
            "note_zh": "若有炉渣、窑炉、工具痕迹，则生产加工用途上升。",
            "note_en": "Production probability rises with slag, kiln/furnace traces, and tool evidence.",
        },
    ])


def bayesian_function_inference(
    context_text: str,
    image_feats: Optional[Dict[str, float]],
    kb: pd.DataFrame,
    lang: str = "zh",
) -> Tuple[pd.DataFrame, List[str]]:
    context = (context_text or "").lower()
    priors = kb[["hypothesis", "prior"]].copy()
    priors["score"] = priors["prior"].astype(float)
    evidence_chain = []

    for i, row in kb.iterrows():
        hyp = row["hypothesis"]
        keyword_hits = sum(1 for kw in str(row["keywords"]).lower().split() if kw in context)
        cue_hits = sum(1 for kw in str(row["context_cues"]).lower().split() if kw in context)
        priors.loc[priors["hypothesis"] == hyp, "score"] += 0.08 * keyword_hits + 0.12 * cue_hits

        if image_feats:
            sym = max(image_feats.get("symmetry_h", 0), image_feats.get("symmetry_v", 0))
            edge = image_feats.get("edge_proxy", 0)
            dark_ratio = image_feats.get("dark_ratio", 0)

            rule = str(row.get("feature_rule", ""))
            if rule == "high_symmetry" and sym > 0.78:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.18
            if rule == "medium_symmetry" and 0.60 <= sym <= 0.82:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.12
            if rule == "low_to_medium_symmetry" and 0.45 <= sym <= 0.75:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.10
            if rule == "low_symmetry" and sym < 0.6:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.15
            if hyp == "production" and edge > 0.18:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.08
            if hyp == "burial_associated" and dark_ratio > 0.55:
                priors.loc[priors["hypothesis"] == hyp, "score"] += 0.05

    scores = priors["score"].clip(lower=1e-6).values
    post = scores / scores.sum()
    priors["posterior"] = post
    priors = priors.sort_values("posterior", ascending=False)

    top3 = priors.head(3)["hypothesis"].tolist()
    for hyp in top3:
        row = kb[kb["hypothesis"] == hyp].iloc[0]
        note = row["note_zh"] if lang == "zh" else row["note_en"]
        evidence_chain.append(f"{hyp}: {note}")

    return priors[["hypothesis", "posterior"]], evidence_chain


# =========================
# Sidebar
# =========================
st.sidebar.header("ArchaeoInfer")
lang_choice = st.sidebar.selectbox(
    tr("sidebar_lang"),
    options=[("zh", "中文"), ("en", "English")],
    format_func=lambda x: x[1],
    index=0 if st.session_state.lang == "zh" else 1,
)
st.session_state.lang = lang_choice[0]

st.title(tr("title"))
st.caption(tr("subtitle"))
with st.expander(tr("disclaimer_title"), expanded=True):
    st.write(tr("disclaimer_body"))

st.sidebar.subheader(tr("sidebar_upload"))
uploaded = st.sidebar.file_uploader("CSV", type=["csv"], help=tr("dataset_help"))
use_demo = st.sidebar.checkbox(tr("use_demo"), value=(uploaded is None))

if uploaded is not None and not use_demo:
    raw_df = pd.read_csv(uploaded)
    st.sidebar.success("CSV loaded")
else:
    raw_df = create_demo_dataset()

df, missing_cols = ensure_recommended_columns(raw_df)
if missing_cols:
    st.warning(f"{tr('missing_cols_warning')} Missing: {', '.join(missing_cols)}")

with st.expander(tr("uploaded_dataset"), expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    tr("tab1"), tr("tab2"), tr("tab3"), tr("tab4"), tr("tab5"), tr("tab6")
])

# Shared defaults
default_env = [c for c in ["dist_to_river", "slope", "soil_quality", "elevation", "resource_access"] if c in df.columns]
default_culture = [c for c in ["dist_to_ancient_route", "dist_to_political_center", "dist_to_ritual_landscape"] if c in df.columns]
default_detect = [c for c in ["ndvi", "soil_moisture", "modern_coverage", "erosion_risk", "season_visibility"] if c in df.columns]

with tab1:
    st.subheader(tr("region_model_title"))
    st.write(tr("region_model_desc"))

    env_features = st.multiselect(tr("env_features"), options=list(df.columns), default=default_env, key="env_features")
    culture_features = st.multiselect(tr("culture_features"), options=list(df.columns), default=default_culture, key="culture_features")
    time_feature = st.selectbox(tr("time_feature"), options=list(df.columns), index=list(df.columns).index("time_weight") if "time_weight" in df.columns else 0)
    target_col = st.selectbox(tr("target_site"), options=list(df.columns), index=list(df.columns).index("is_site") if "is_site" in df.columns else 0)

    if st.button(tr("run_model"), key="run_exists"):
        exists_df, coef_map = compute_exists_probability(df, env_features, culture_features, time_feature, target_col)
        st.session_state.exists_df = exists_df
        st.session_state.exists_coef_map = coef_map
        st.success(tr("exists_prob_done"))

    if "exists_df" in st.session_state:
        exists_df = st.session_state.exists_df
        coef_map = st.session_state.exists_coef_map
        st.dataframe(exists_df[[c for c in ["x", "y", "p_exists", target_col] if c in exists_df.columns]].head(20), use_container_width=True)
        st.write("**Coefficients / 系数**")
        st.json({k: round(v, 4) for k, v in coef_map.items()})
        st.write(f"**{tr('top_zones')}**")
        top_exists = exists_df.sort_values("p_exists", ascending=False).head(10)
        st.dataframe(top_exists[[c for c in ["x", "y", "p_exists", "culture_type"] if c in top_exists.columns]], use_container_width=True)
        st.caption(tr("show_map_note"))
        st.scatter_chart(exists_df[["x", "y", "p_exists"]].rename(columns={"x": "x", "y": "y", "p_exists": "z"}), x="x", y="y")

with tab2:
    st.subheader(tr("detect_model_title"))
    st.write(tr("detect_model_desc"))

    detect_features = st.multiselect(tr("detect_features"), options=list(df.columns), default=default_detect, key="detect_features")

    if st.button(tr("run_model"), key="run_detect"):
        source_df = st.session_state.get("exists_df", df.copy())
        detect_df, detect_weights = compute_detectability(source_df, detect_features)
        detect_df = combine_observed_probability(detect_df) if "p_exists" in detect_df.columns else detect_df
        st.session_state.detect_df = detect_df
        st.session_state.detect_weights = detect_weights
        st.success(tr("detect_prob_done"))
        if "p_observed" in detect_df.columns:
            st.success(tr("combined_prob_done"))

    if "detect_df" in st.session_state:
        detect_df = st.session_state.detect_df
        st.write("**Weights / 权重**")
        st.json(st.session_state.detect_weights)
        cols_to_show = [c for c in ["x", "y", "p_detectable", "p_exists", "p_observed"] if c in detect_df.columns]
        st.dataframe(detect_df[cols_to_show].head(20), use_container_width=True)
        if "p_observed" in detect_df.columns:
            st.latex(r"P(\mathrm{observed}) = P(\mathrm{exists}) \times P(\mathrm{detectable})")
            st.write(f"**{tr('top_zones')}**")
            top_obs = detect_df.sort_values("p_observed", ascending=False).head(10)
            st.dataframe(top_obs[[c for c in ["x", "y", "p_observed", "p_exists", "p_detectable"] if c in top_obs.columns]], use_container_width=True)
        st.scatter_chart(detect_df[["x", "y", "p_detectable"]].rename(columns={"p_detectable": "z"}), x="x", y="y")

with tab3:
    st.subheader(tr("structure_title"))
    st.write(tr("structure_desc"))

    st.markdown(f"**{tr('structure_settings')}**")
    x_col = st.selectbox(tr("coord_x"), options=list(df.columns), index=list(df.columns).index("x") if "x" in df.columns else 0)
    y_col = st.selectbox(tr("coord_y"), options=list(df.columns), index=list(df.columns).index("y") if "y" in df.columns else 0)
    unit_col = st.selectbox(tr("unit_type"), options=list(df.columns), index=list(df.columns).index("unit_type") if "unit_type" in df.columns else 0)

    culture_options = sorted(df["culture_type"].astype(str).unique().tolist()) if "culture_type" in df.columns else ["Generic"]
    selected_culture = st.selectbox(tr("culture_type"), options=culture_options)
    symmetry_strength = st.slider(tr("symmetry_strength"), 0.0, 1.0, 0.7, 0.05)
    radius = st.slider(tr("neighbor_radius"), 1.0, 30.0, 8.0, 1.0)

    if st.button(tr("run_model"), key="run_structure"):
        if x_col not in df.columns or y_col not in df.columns:
            st.error(tr("not_enough_cols"))
        else:
            source_df = st.session_state.get("detect_df", st.session_state.get("exists_df", df.copy()))
            suggestions = structural_completion(source_df, x_col, y_col, unit_col, selected_culture, symmetry_strength, radius)
            st.session_state.structure_df = suggestions

    if "structure_df" in st.session_state:
        structure_df = st.session_state.structure_df
        st.write(f"**{tr('suggested_units')}**")
        st.dataframe(structure_df, use_container_width=True)
        if not structure_df.empty:
            chart_df = structure_df.rename(columns={"suggested_x": "x", "suggested_y": "y"})
            st.scatter_chart(chart_df[["x", "y", "score"]].rename(columns={"score": "z"}), x="x", y="y")

with tab4:
    st.subheader(tr("burial_title"))
    st.write(tr("burial_desc"))

    if st.button(tr("run_model"), key="run_burial"):
        source_df = st.session_state.get("detect_df", st.session_state.get("exists_df", df.copy()))
        burial_df = estimate_depth_and_age(source_df)
        st.session_state.burial_df = burial_df

    if "burial_df" in st.session_state:
        burial_df = st.session_state.burial_df
        st.write(f"**{tr('depth_age_output')}**")
        cols = [c for c in ["x", "y", "depth_est_m", "depth_low_m", "depth_high_m", "estimated_age_bp", "age_low_bp", "age_high_bp"] if c in burial_df.columns]
        st.dataframe(burial_df[cols].head(30), use_container_width=True)
        st.bar_chart(burial_df[["depth_est_m"]].head(30))

with tab5:
    st.subheader(tr("image_title"))
    st.write(tr("image_desc"))

    image_file = st.file_uploader(tr("upload_image"), type=["png", "jpg", "jpeg"], key="img_upload")
    if image_file is not None:
        image_bytes = image_file.read()
        st.image(image_bytes, width=320)
        img_feats = extract_basic_image_features(image_bytes)
        st.write(f"**{tr('image_features')}**")
        st.json(img_feats)
    else:
        img_feats = None
        st.info(tr("no_image_uploaded"))

    context_text = st.text_area(tr("context_input"), placeholder=tr("context_placeholder"), height=140)
    kb_file = st.file_uploader(tr("kb_upload"), type=["csv"], key="kb_upload")

    if kb_file is not None:
        kb = pd.read_csv(kb_file)
    else:
        kb = default_knowledge_base()
        st.caption(tr("default_kb"))

    if st.button(tr("bayes_run"), key="run_bayes"):
        post_df, evidence_chain = bayesian_function_inference(context_text, img_feats, kb, lang=st.session_state.lang)
        st.session_state.posterior_df = post_df
        st.session_state.evidence_chain = evidence_chain
        st.session_state.image_features = img_feats or {}

    if "posterior_df" in st.session_state:
        st.write(f"**{tr('usage_hypotheses')}**")
        post_df = st.session_state.posterior_df.copy()
        post_df["posterior"] = post_df["posterior"].round(4)
        st.dataframe(post_df, use_container_width=True)
        st.bar_chart(post_df.set_index("hypothesis"))
        st.write(f"**{tr('evidence_chain')}**")
        for item in st.session_state.evidence_chain:
            st.write(f"- {item}")

with tab6:
    st.subheader(tr("download_title"))
    st.write(tr("download_desc"))

    outputs = {}
    for key in ["exists_df", "detect_df", "structure_df", "burial_df", "posterior_df"]:
        if key in st.session_state:
            val = st.session_state[key]
            if isinstance(val, pd.DataFrame):
                outputs[key] = val

    if outputs:
        selected_output = st.selectbox("Output", options=list(outputs.keys()))
        out_df = outputs[selected_output]
        st.dataframe(out_df.head(50), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(tr("download_csv"), data=csv_bytes, file_name=f"{selected_output}.csv", mime="text/csv")

        meta = {
            "language": st.session_state.lang,
            "available_outputs": list(outputs.keys()),
            "notes": "Prototype outputs for ArchaeoInfer.",
        }
        st.download_button(
            tr("download_json"),
            data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="archaeoinfer_meta.json",
            mime="application/json",
        )
    else:
        st.info("No outputs yet. Run one or more modules first.")
