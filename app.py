from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cars.csv"
MODELS_DIR = ROOT / "models"
TARGET = "Price(euro)"

MODEL_TITLES = {
    "ML1_polynomial_ridge": "ML1: полиномиальная регрессия, степень 2",
    "ML2_gradient_boosting": "ML2: ансамблевая модель, бустинг",
    "ML3_catboost": "ML3: CatBoostRegressor",
    "ML4_bagging": "ML4: ансамблевая модель, бэггинг",
    "ML5_stacking": "ML5: ансамблевая модель, стэкинг",
    "ML6_mlp_neural_network": "ML6: глубокая полносвязная нейронная сеть",
}


st.set_page_config(page_title="РГР: инференс ML-моделей", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if not path.exists():
        return {"features": [c for c in load_data().columns if c != TARGET], "metrics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_name: str):
    if model_name == "ML3_catboost":
        model = CatBoostRegressor()
        model.load_model(str(MODELS_DIR / "ML3_catboost.cbm"))
        return model
    with (MODELS_DIR / f"{model_name}.pkl").open("rb") as file:
        return pickle.load(file)


def one_hot_choice(prefix: str, choice: str, features: list[str]) -> dict[str, int]:
    return {feature: int(feature == f"{prefix}_{choice}") for feature in features if feature.startswith(f"{prefix}_")}


def decode_options(features: list[str], prefix: str) -> list[str]:
    return sorted(feature.replace(f"{prefix}_", "") for feature in features if feature.startswith(f"{prefix}_"))


def build_manual_input(features: list[str], df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Ручной ввод параметров автомобиля")
    num_cols = ["Distance", "Engine_capacity(cm3)", "Age"]
    fuel_options = decode_options(features, "Fuel_type")
    transmission_options = decode_options(features, "Transmission")
    make_options = decode_options(features, "Make")
    style_options = decode_options(features, "Style")

    c1, c2, c3 = st.columns(3)
    with c1:
        distance = st.number_input(
            "Пробег, км", min_value=0, max_value=1_500_000,
            value=int(df["Distance"].median()), step=1_000
        )
    with c2:
        engine = st.number_input(
            "Объем двигателя, см3", min_value=0, max_value=10_000,
            value=int(df["Engine_capacity(cm3)"].median()), step=100
        )
    with c3:
        age = st.number_input(
            "Возраст автомобиля, лет", min_value=0, max_value=100,
            value=int(df["Age"].median()), step=1
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        fuel = st.selectbox("Тип топлива", fuel_options, index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0)
    with c5:
        transmission = st.selectbox("Коробка передач", transmission_options)
    with c6:
        make = st.selectbox("Марка", make_options, index=make_options.index("Volkswagen") if "Volkswagen" in make_options else 0)
    with c7:
        style = st.selectbox("Кузов", style_options, index=style_options.index("Sedan") if "Sedan" in style_options else 0)

    row = {feature: 0 for feature in features}
    row.update({"Distance": distance, "Engine_capacity(cm3)": engine, "Age": age})
    row.update(one_hot_choice("Fuel_type", fuel, features))
    row.update(one_hot_choice("Transmission", transmission, features))
    row.update(one_hot_choice("Make", make, features))
    row.update(one_hot_choice("Style", style, features))
    return pd.DataFrame([row], columns=features)


def validate_uploaded(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    errors = []
    missing = [col for col in features if col not in df.columns]
    if missing:
        errors.append(f"В CSV отсутствуют признаки: {', '.join(missing[:8])}" + ("..." if len(missing) > 8 else ""))
    if errors:
        return None, errors
    prepared = df[features].copy()
    for col in prepared.columns:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    if prepared.isna().any().any():
        errors.append("В признаках есть нечисловые или пустые значения.")
    return prepared, errors


def page_developer() -> None:
    st.title("РГР: инференс моделей машинного обучения")
    left, right = st.columns([1, 2])
    with left:
        photo = ROOT / "assets" / "developer_photo.jpg"
        if photo.exists():
            st.image(str(photo), width=400)
    with right:
        st.markdown(
            """
            **Разработчик моделей ML:** Беньковский Максим Дмитриевич\n
            **Учебная группа:** ФИТ-242\n
            **Тема РГР:** "Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных"
            """
        )


def page_dataset(df: pd.DataFrame) -> None:
    st.title("Набор данных о стоимости машин на вторичном рынке в Молдове")
    st.write(
        "Датасет описывает автомобили на вторичном рынке. Целевая переменная `Price(euro)` "
        "отражает стоимость автомобиля в евро. Признаки включают пробег, объем двигателя, возраст, "
        "тип топлива, коробку передач, марку и тип кузова."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", f"{len(df):,}".replace(",", " "))
    c2.metric("Признаков", 9)
    c3.metric("Пропусков", int(df.isna().sum().sum()))
    c4.metric("Медианная цена", f"{df[TARGET].median():,.0f} €".replace(",", " "))

    st.subheader("Предобработка")
    st.write(
        "В процессе EDA из датасета были удалены записи с неестественными значениями (пробег > 1000000 км), были удалены дубликаты, " \
        "категориальные признаки были представлены one-hot кодированием. Все признаки числовые, "
        "пропуски отсутствуют. Для моделей с чувствительностью к масштабу используется стандартизация; "
        "для первой модели дополнительно формируются полиномиальные признаки степени 2."
    )
    st.dataframe(df.head(30), use_container_width=True)
    st.subheader("Сводная статистика числовых признаков")
    st.dataframe(df[["Distance", "Engine_capacity(cm3)", "Age", TARGET]].describe(), use_container_width=True)


def page_visualizations(df: pd.DataFrame) -> None:
    st.title("Визуализации зависимостей")
    sample = df.sample(min(3500, len(df)), random_state=42)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matplotlib: цена и возраст")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample["Age"], sample[TARGET], s=12, alpha=0.35)
        ax.set_xlabel("Возраст, лет")
        ax.set_ylabel("Цена, евро")
        st.pyplot(fig)
    with c2:
        st.subheader("Seaborn: распределение цены")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(sample[TARGET], bins=45, kde=True, ax=ax)
        ax.set_xlabel("Цена, евро")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Seaborn: корреляции")
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df[["Distance", "Engine_capacity(cm3)", "Age", TARGET]].corr()
        sns.heatmap(corr, annot=True, cmap="RdYlBu_r", ax=ax)
        st.pyplot(fig)
    with c4:
        st.subheader("Plotly: цена по типу топлива")
        fuel_cols = [c for c in df.columns if c.startswith("Fuel_type_")]
        fuel_df = df[fuel_cols + [TARGET]].copy()
        fuel_df["Fuel"] = fuel_df[fuel_cols].idxmax(axis=1).str.replace("Fuel_type_", "")
        fig = px.box(fuel_df, x="Fuel", y=TARGET, points=False, labels={TARGET: "Цена, евро", "Fuel": "Топливо"})
        st.plotly_chart(fig, use_container_width=True)



def page_prediction(df: pd.DataFrame, metadata: dict) -> None:
    st.title("Инференс моделей")
    features = metadata["features"]
    metrics = metadata.get("metrics", {})

    if not (MODELS_DIR / "metadata.json").exists():
        st.error("Модели еще не обучены. Запустите команду `python train_models.py`, затем обновите страницу.")
        return

    metric_rows = []
    for name, values in metrics.items():
        metric_rows.append({
            "Модель": MODEL_TITLES.get(name, name),
            "R²": round(values["r2"], 4),
            "MAE, евро": round(values["mae"], 2),
            "RMSE, евро": round(values["rmse"], 2),
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    model_name = st.selectbox(
        "Выберите модель для предсказания",
        list(MODEL_TITLES.keys()),
        format_func=lambda key: MODEL_TITLES[key],
        index=2 if "ML3_catboost" in MODEL_TITLES else 0,
    )
    mode = st.radio("Источник данных", ["Ручной ввод", "Загрузка CSV"], horizontal=True)

    input_df = None
    if mode == "Ручной ввод":
        input_df = build_manual_input(features, df)
    else:
        uploaded = st.file_uploader("Загрузите CSV с теми же признаками, что и cars.csv", type=["csv"])
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            input_df, errors = validate_uploaded(uploaded_df, features)
            for error in errors:
                st.error(error)
            if input_df is not None:
                st.success(f"Файл принят: {len(input_df)} строк.")

    if input_df is not None:
        with st.expander("Проверить подготовленные признаки"):
            st.dataframe(input_df.head(20), use_container_width=True)

        if st.button("Получить прогноз", type="primary"):
            model = load_model(model_name)
            predictions = pd.Series(model.predict(input_df)).clip(lower=0)
            if len(predictions) == 1:
                st.success(f"Прогнозируемая стоимость автомобиля: {predictions.iloc[0]:,.0f} €".replace(",", " "))
            else:
                result = input_df.copy()
                result["Predicted_Price(euro)"] = predictions.round(0).astype(int)
                st.dataframe(result[["Predicted_Price(euro)"]].join(input_df.head(len(result))), use_container_width=True)
                st.download_button(
                    "Скачать прогнозы CSV",
                    data=result.to_csv(index=False).encode("utf-8-sig"),
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                )


def main() -> None:
    df = load_data()
    metadata = load_metadata()
    page = st.sidebar.radio(
        "Навигация",
        [
            "1. Разработчик",
            "2. Датасет и EDA",
            "3. Визуализации",
            "4. Предсказание",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Целевая переменная: Price(euro)")

    if page.startswith("1"):
        page_developer()
    elif page.startswith("2"):
        page_dataset(df)
    elif page.startswith("3"):
        page_visualizations(df)
    else:
        page_prediction(df, metadata)


if __name__ == "__main__":
    main()
