import streamlit as st
import pandas as pd

st.set_page_config(page_title="Crypto Dashboard")

df = pd.read_json("https://api.binance.com/api/v3/ticker/24hr")

crpytoList = {
    "Price 1": "BTCBUSD",
    "Price 2": "ETHBUSD",
    "Price 3": "BNBBUSD",
}

col1, col2, col3 = st.columns(3)

for i in range(len(crpytoList.keys())):
    selected_crypto_label = list(crpytoList.keys())[i]
    selected_crypto_index = list(df.symbol).index(crpytoList[selected_crypto_label])
    selected_crypto = st.sidebar.selectbox(
        selected_crypto_label, df.symbol, selected_crypto_index, key=str(i)
    )
    col_df = df[df.symbol == selected_crypto]
    col_price = col_df.weightedAvgPrice
    col_percent = f"{float(col_df.priceChangePercent)}%"
    with col1:
        st.metric(selected_crypto, col_price, col_percent)


st.header("")
