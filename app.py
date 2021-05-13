import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd


@st.cache
def load_model(model_path="./model/model.cbm"):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


model = load_model("./model/model.cbm")
df = pd.read_csv("./data/model_input.csv")
bairros = df["crawler"].unique()

st.title("Calculadora de imóveis")
st.subheader("Entre com as características do seu imóvel")

bairro = st.selectbox(
    "Nome do bairro",
    options=bairros,
)

area = st.number_input(
    label="Área do apto", min_value=10, max_value=500, value=70, step=25
)

condominio = st.number_input(
    label="Valor do condomínio", min_value=100, max_value=5000, value=1000, step=100
)

quartos = st.slider(label="# quartos", min_value=1, max_value=5, value=2)
banheiros = st.slider(label="# banheiros", min_value=1, max_value=5, value=2)
garagens = st.slider(label="# garagens", min_value=0, max_value=5, value=2)

preco = model.predict([area, condominio, quartos, banheiros, garagens, bairro])

if st.button(label="Calcular"):
    st.subheader(f"Preço estimado: R$ {preco:,.2f}")
    with open("memoria.csv", "a") as f:
        f.writelines(
            f"{bairro},{area},{condominio},{quartos},{banheiros},{garagens},{preco}\n"
        )
