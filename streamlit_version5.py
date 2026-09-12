import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime
from FinMind.data import DataLoader

st.set_page_config(page_title="ETF 損益分析系統", layout="wide")
st.title("📈 ETF 損益分析與預測系統")
st.markdown("---")

fm_api = DataLoader()
api = fm_api

# ============================================================
# Version 5 AI 核心模組
# KMeans 市場狀態 + FinBERT 新聞情緒 + TCN 多步 OHLC
# ============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from datetime import date
from dateutil.relativedelta import relativedelta
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, Add
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_SEED = 42
LOOKBACK_DAYS = 60
FORECAST_DAYS = 22
TRAIN_EPOCHS = 80
NEWS_MAX_DISPLAY = 30
NEWS_RECENCY_DECAY = 0.05
BATCH_SIZE = 32
KMEANS_CLUSTERS = 3
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

@st.cache_resource(show_spinner=False)
def get_sentiment_classifier():
    return pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone-chinese",
        tokenizer="yiyanghkust/finbert-tone-chinese",
        truncation=True,
        max_length=128
    )

sentiment_classifier = None
def ensure_sentiment_classifier():
    global sentiment_classifier
    if sentiment_classifier is None:
        sentiment_classifier = get_sentiment_classifier()
    return sentiment_classifier



@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        rate_data = yf.Ticker("USDTWD=X").history(period="1d")
        if not rate_data.empty:
            last_close = rate_data['Close'].iloc[-1]
            if pd.notna(last_close) and last_close > 0:
                return float(last_close), False
        return 32.0, True
    except:
        return 32.0, True

current_usd_twd_rate, is_rate_default = get_exchange_rate()

# ==========================================
# 頂層三大分頁架構
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 個股歷史趨勢與技術分析", "💰 定期定額回測", "📺 市場趨勢即時掃描", "🤖 AI 多模態月度預測"])

# ------------------------------------------
# 🧠 第一分頁：單檔股票技術分析
# ------------------------------------------
with tab1:
    with st.container(border=True):
        st.subheader("🛠️ 查詢與條件設定")
        t1_c1, t1_c2, t1_c3 = st.columns([1.5, 1.5, 2])
        
        with t1_c1:
            preset_options = ["0050.TW (元大台灣50)", "0056.TW (元大高股息)", "2330.TW (台積電)", "AAPL (蘋果)", "🔍 自訂輸入代碼..."]
            selected_preset = st.selectbox("選擇查詢標的：", preset_options, key="t1_preset")
            if selected_preset == "🔍 自訂輸入代碼...":
                etf_option = st.text_input("請輸入代碼 (台股請加 .TW)：", "00940.TW", key="t1_custom")
            else:
                etf_option = selected_preset.split(" ")[0]
            data_source = st.radio("優先資料來源：", ("Yahoo Finance", "FinMind (僅限台股)"), horizontal=True, key="t1_source")
            
        with t1_c2:
            quick_time = st.radio("時間區間：", ["近 1 個月", "近 3 個月", "今年以來", "全部歷史", "進階自訂..."], horizontal=True, key="t1_time")
            is_custom_date = False
            
            if quick_time == "近 1 個月": days_to_fetch = 30
            elif quick_time == "近 3 個月": days_to_fetch = 90
            elif quick_time == "今年以來": 
                days_to_fetch = (datetime.date.today() - datetime.date(datetime.date.today().year, 1, 1)).days
            elif quick_time == "全部歷史": days_to_fetch = 365*20
            else:
                advanced_time = st.selectbox("進階區間選擇：", ["近一週", "近半年", "近兩年", "📅 自訂日曆區間"], key="t1_adv_time")
                if advanced_time == "近一週": days_to_fetch = 7
                elif advanced_time == "近半年": days_to_fetch = 180
                elif advanced_time == "近兩年": days_to_fetch = 365*2
                else:
                    date_range = st.date_input("選擇起訖日期", [datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()], key="t1_date")
                    if len(date_range) == 2:
                        custom_start, custom_end = date_range
                        is_custom_date = True
                    else:
                        st.stop()
            
        with t1_c3:
            currency_label = st.radio("顯示計價幣別：", ("預設 (新台幣)", "美元 (USD)"), horizontal=True, key="t1_currency")
            st.markdown("**主圖表顯示控制：**")
            chart_type = st.radio("主圖表類型", ["📈 收盤價折線圖", "📊 專業 K 線與成交量圖"], horizontal=True, key="t1_chart_type")

        search_button = st.button("🚀 載入歷史數據並繪圖", use_container_width=True, key="t1_btn")

    if search_button:
        with st.spinner(f"系統正撈取 {etf_option} 歷史數據中..."):
            hist_data = pd.DataFrame()
            fetch_success = False
            actual_source = data_source
            
            if is_custom_date:
                end_date = custom_end
                display_start_date = custom_start
            else:
                end_date = datetime.date.today()
                display_start_date = end_date - datetime.timedelta(days=days_to_fetch)
            
            fetch_start_date = display_start_date - datetime.timedelta(days=90)

            try:
                if data_source == "Yahoo Finance":
                    temp_data = yf.Ticker(etf_option).history(start=fetch_start_date, end=end_date, auto_adjust=True)
                    # 👇 加上這行：無情刪除沒有收盤價的 NaN 幽靈數據
                    temp_data = temp_data.dropna(subset=['Close']) 
                    
                    if not temp_data.empty and len(temp_data) > 1:
                        hist_data = temp_data
                        fetch_success = True
                    else:
                        if etf_option.endswith(".TW"):
                            actual_source = "FinMind (備援)"
                            fm_id = etf_option.replace(".TW", "")
                            fm_df = fm_api.taiwan_stock_daily(stock_id=fm_id, start_date=fetch_start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
                            if not fm_df.empty:
                                fm_df = fm_df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
                                fm_df['Date'] = pd.to_datetime(fm_df['Date'])
                                hist_data = fm_df.set_index('Date')
                                fetch_success = True
                else:
                    fm_id = etf_option.replace(".TW", "")
                    fm_df = fm_api.taiwan_stock_daily(stock_id=fm_id, start_date=fetch_start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
                    if not fm_df.empty:
                        fm_df = fm_df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
                        fm_df['Date'] = pd.to_datetime(fm_df['Date'])
                        hist_data = fm_df.set_index('Date')
                        fetch_success = True
            except Exception:
                pass

            if fetch_success:
                if hist_data.index.tz is not None:
                    hist_data.index = hist_data.index.tz_localize(None)

                is_tw_stock = etf_option.endswith(".TW") or "FinMind" in actual_source
                if currency_label == "預設 (新台幣)":
                    if not is_tw_stock: 
                        for col in ['Open', 'High', 'Low', 'Close']:
                            if col in hist_data.columns: hist_data[col] = hist_data[col] * current_usd_twd_rate
                    currency_symbol = "NT$"
                else:
                    if is_tw_stock: 
                        for col in ['Open', 'High', 'Low', 'Close']:
                            if col in hist_data.columns: hist_data[col] = hist_data[col] / current_usd_twd_rate
                    currency_symbol = "US$"

                hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['MA60'] = hist_data['Close'].rolling(window=60).mean()
                
                delta = hist_data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=13, adjust=False).mean()
                avg_loss = loss.ewm(com=13, adjust=False).mean()
                rs = avg_gain / avg_loss
                hist_data['RSI'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
                
                hist_data['Price_Diff'] = hist_data['Close'].diff()
                hist_data['Pct_Change'] = hist_data['Close'].pct_change() * 100
                hist_data['Hover_Text'] = (
                    "<b>價格差:</b> " + hist_data['Price_Diff'].apply(lambda x: f"{x:+.2f}" if pd.notnull(x) else "0") + "<br>" +
                    "<b>漲跌幅:</b> " + hist_data['Pct_Change'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "0%")
                )

                if quick_time != "全部歷史":
                    hist_data = hist_data[hist_data.index >= pd.Timestamp(display_start_date)]

                st.session_state['t1_data'] = hist_data
                st.session_state['t1_meta'] = {'etf': etf_option, 'sym': currency_symbol, 'src': actual_source}
            else:
                st.error("❌ 無法獲取資料，請檢查代碼或網路。")

    if st.session_state.get('t1_data') is not None:
        data = st.session_state['t1_data']
        meta = st.session_state['t1_meta']
        latest = data.iloc[-1]
        
        st.success(f"✅ 成功自 {meta['src']} 載入 **{meta['etf']}**！")
        
        c1, c2, c3 = st.columns(3)
        price_diff = latest['Close'] - data['Close'].iloc[0]
        with c1: st.metric("期間最後結算價格", f"{meta['sym']}{latest['Close']:.2f}")
        with c2: st.metric("選定區間總漲跌幅", f"{price_diff:+.2f}", f"{(price_diff/data['Close'].iloc[0]*100):+.2f}%")
        with c3: st.metric("顯示歷史天數", f"{len(data)} 天")

        with st.expander("⚙️ 展開進階圖表與指標設定 (MA均線、RSI)"):
            exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.5])
            with exp_c1:
                st.markdown("**均線設定**")
                t1_show_ma20 = st.toggle("顯示 MA20 (月線)", value=True, key="t1_tg_ma20")
                t1_show_ma60 = st.toggle("顯示 MA60 (季線)", value=True, key="t1_tg_ma60")
                if st.session_state.get('t1_chart_type') == "📈 收盤價折線圖":
                    show_markers = st.toggle("顯示折線圖資料點 (圓點)", value=False, key="t1_markers")
            with exp_c2:
                st.markdown("**副圖設定**")
                show_rsi = st.toggle("顯示 RSI 技術指標圖", value=False, key="t1_tg_rsi")
                if show_rsi:
                    show_rsi_markers = st.toggle("顯示 RSI 資料點", value=False, key="t1_tg_rsi_mk")
            with exp_c3:
                if show_rsi:
                    st.markdown("**🎨 RSI 顏色自訂**")
                    color_col1, color_col2, color_col3 = st.columns(3)
                    with color_col1: color_overbought = st.color_picker("超買區 (>70)", "#FF0000", key="t1_cp_ob")
                    with color_col2: color_normal = st.color_picker("RSI 主線", "#800080", key="t1_cp_nm")
                    with color_col3: color_oversold = st.color_picker("超賣區 (<30)", "#008000", key="t1_cp_os")
                else:
                    st.markdown("**🎨 RSI 顏色自訂**")
                    st.caption("開啟左側 RSI 副圖後即可設定")
        
        required_kline_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        has_full_kline_data = all(col in data.columns for col in required_kline_cols)

        actual_dates = data.index.strftime("%Y-%m-%d").tolist()
        dt_all = pd.date_range(start=data.index[0], end=data.index[-1], freq='D').strftime("%Y-%m-%d").tolist()
        dt_breaks = list(set(dt_all) - set(actual_dates))

        if st.session_state.get('t1_chart_type') == "📊 專業 K 線與成交量圖":
            if has_full_kline_data:
                with st.container(border=True):
                    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], row_titles=["價格", "成交量"])
                    fig_k.add_trace(go.Candlestick(
                        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                        name='K線條', increasing_line_color='red', increasing_fillcolor='red', decreasing_line_color='green', decreasing_fillcolor='green'
                    ), row=1, col=1)
                    
                    if t1_show_ma20: fig_k.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
                    if t1_show_ma60: fig_k.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name='MA60', line=dict(color='blue', width=1.5)), row=1, col=1)
                    
                    bar_colors = ['red' if row['Close'] >= row['Open'] else 'green' for index, row in data.iterrows()]
                    fig_k.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=bar_colors, opacity=0.7), row=2, col=1)
                    
                    fig_k.update_layout(
                        title=f"{meta['etf']} 歷史 K 線與量能圖", 
                        xaxis_rangeslider_visible=False, 
                        height=600, 
                        template="plotly_white", 
                        hovermode="x unified",
                        xaxis=dict(rangebreaks=[dict(values=dt_breaks)])
                    )
                    st.plotly_chart(fig_k, use_container_width=True)
            else:
                st.warning(f"⚠️ {meta['etf']} 缺乏完整的開高低收或成交量資料，無法繪製 K 線，已自動降級顯示折線圖。")
                with st.container(border=True):
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=data.index, y=data['Close'], 
                        mode='lines+markers' if st.session_state.get('t1_markers') else 'lines',
                        name='收盤價', text=data['Hover_Text'], hovertemplate="<b>時間</b>: %{x}<br><b>價格</b>: %{y:.2f}<br>%{text}<extra></extra>"
                    ))
                    if t1_show_ma20: fig_line.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(dash='dot', color='orange')))
                    if t1_show_ma60: fig_line.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name='MA60', line=dict(dash='dot', color='green')))
                    fig_line.update_layout(
                        title=f"{meta['etf']} 歷史收盤價走勢 (降級顯示)", 
                        xaxis_title="日期", yaxis_title=f"價格 ({meta['sym']})", 
                        height=450, template="plotly_white", hovermode="x unified",
                        xaxis=dict(rangebreaks=[dict(values=dt_breaks)])
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
        else:
            with st.container(border=True):
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=data.index, y=data['Close'], 
                    mode='lines+markers' if st.session_state.get('t1_markers') else 'lines',
                    name='收盤價', text=data['Hover_Text'], hovertemplate="<b>時間</b>: %{x}<br><b>價格</b>: %{y:.2f}<br>%{text}<extra></extra>"
                ))
                if t1_show_ma20: fig_line.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(dash='dot', color='orange')))
                if t1_show_ma60: fig_line.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name='MA60', line=dict(dash='dot', color='green')))
                fig_line.update_layout(
                    title=f"{meta['etf']} 歷史收盤價走勢", 
                    xaxis_title="日期", yaxis_title=f"價格 ({meta['sym']})", 
                    height=450, template="plotly_white", hovermode="x unified",
                    xaxis=dict(rangebreaks=[dict(values=dt_breaks)])
                )
                st.plotly_chart(fig_line, use_container_width=True)

        if show_rsi:
            with st.container(border=True):
                fig_rsi = go.Figure()
                rsi_mode = 'lines+markers' if st.session_state.get('t1_tg_rsi_mk') else 'lines'

                def hex_to_rgba(hex_color, opacity=0.3):
                    hex_color = hex_color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    return f'rgba({r}, {g}, {b}, {opacity})'

                valid_rsi = data.dropna(subset=['RSI'])
                if not valid_rsi.empty:
                    x_orig = valid_rsi.index
                    y_orig = valid_rsi['RSI'].values
                    x_fill, y_fill = [], []
                    
                    for i in range(len(x_orig) - 1):
                        x1, y1 = x_orig[i], y_orig[i]
                        x2, y2 = x_orig[i+1], y_orig[i+1]
                        x_fill.append(x1)
                        y_fill.append(y1)
                        
                        if (y1 < 70 and y2 > 70) or (y1 > 70 and y2 < 70):
                            frac = (70 - y1) / (y2 - y1)
                            x_fill.append(x1 + (x2 - x1) * frac)
                            y_fill.append(70.0)
                            
                        if (y1 < 30 and y2 > 30) or (y1 > 30 and y2 < 30):
                            frac = (30 - y1) / (y2 - y1)
                            x_fill.append(x1 + (x2 - x1) * frac)
                            y_fill.append(30.0)
                            
                    x_fill.append(x_orig[-1])
                    y_fill.append(y_orig[-1])
                    
                    x_fill, y_fill = np.array(x_fill), np.array(y_fill)
                    rsi_ob_clamped = np.where(y_fill >= 70, y_fill, 70)
                    rsi_os_clamped = np.where(y_fill <= 30, y_fill, 30)

                    fig_rsi.add_trace(go.Scatter(x=x_fill, y=[70]*len(x_fill), mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                    fig_rsi.add_trace(go.Scatter(x=x_fill, y=rsi_ob_clamped, mode='lines', fill='tonexty', fillcolor=hex_to_rgba(color_overbought), line=dict(width=0), hoverinfo='skip', showlegend=False))
                    fig_rsi.add_trace(go.Scatter(x=x_fill, y=[30]*len(x_fill), mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                    fig_rsi.add_trace(go.Scatter(x=x_fill, y=rsi_os_clamped, mode='lines', fill='tonexty', fillcolor=hex_to_rgba(color_oversold), line=dict(width=0), hoverinfo='skip', showlegend=False))
                    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode=rsi_mode, name='RSI', line=dict(color=color_normal, width=2)))

                fig_rsi.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="超買區 (70)", annotation_position="top left")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="gray", annotation_text="超賣區 (30)", annotation_position="bottom left")
                fig_rsi.update_layout(
                    title=f"{meta['etf']} RSI 相對強弱指標面板", 
                    height=350, template="plotly_white", yaxis_range=[0, 100], hovermode="x unified",
                    xaxis=dict(rangebreaks=[dict(values=dt_breaks)])
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

# ------------------------------------------
# 🧠 第二分頁：獨立的定期定額回測 (升級雙軌數據分流)
# ------------------------------------------
with tab2:
    with st.container(border=True):
        st.subheader("💵 定期定額投資試算 (獨立設定)")
        t2_c1, t2_c2, t2_c3, t2_c4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with t2_c1:
            t2_target = st.text_input("輸入回測標的代碼：", "0050.TW", key="t2_target")
        with t2_c2:
            t2_amount = st.number_input("每月固定投入金額 (NT$)", min_value=1000, value=10000, step=1000, key="t2_amt")
        with t2_c3:
            t2_start_date = st.date_input("選擇回測起始日期", datetime.date(2020, 1, 1), key="t2_date")
        with t2_c4:
            # 🌟 核心增強：加入資料來源選擇，預設為具備股價還原功能的 Yahoo Finance
            t2_source = st.radio("優先資料來源設定：", ("Yahoo Finance", "FinMind (僅限台股)"), horizontal=True, key="t2_source")
            
        if t2_source == "FinMind (僅限台股)":
            st.caption("⚠️ 提示：FinMind 提供未經除權息調整之歷史原價。長期回測建議切換為 **Yahoo Finance**（具備自動還原股價功能），計算總報酬率與真實 ROI 才會精準。")
            
        t2_button = st.button("🚀 執行定期定額回測", use_container_width=True, key="t2_btn")

    if t2_button:
        with st.spinner(f"正在抓取 {t2_target} 歷史數據進行回測..."):
            try:
                start_date_str = t2_start_date.strftime("%Y-%m-%d")
                end_date_str = datetime.date.today().strftime("%Y-%m-%d")
                
                actual_t2_source = t2_source
                bt_data = pd.DataFrame()
                
                # 🌟 核心增強：根據使用者選取的來源進行智慧路由與備援
                if t2_source == "Yahoo Finance":
                    bt_data = yf.Ticker(t2_target).history(start=start_date_str, end=end_date_str, auto_adjust=True)
                    bt_data = bt_data.dropna(subset=['Close']) 
                    if (bt_data.empty or len(bt_data) <= 1) and t2_target.endswith(".TW"):
                        actual_t2_source = "FinMind (備援)"
                        fm_id = t2_target.replace(".TW", "")
                        bt_df = fm_api.taiwan_stock_daily(stock_id=fm_id, start_date=start_date_str, end_date=end_date_str)
                        if not bt_df.empty:
                            bt_df = bt_df.rename(columns={'date': 'Date', 'close': 'Close'})
                            bt_df['Date'] = pd.to_datetime(bt_df['Date'])
                            bt_data = bt_df.set_index('Date')
                else:
                    fm_id = t2_target.replace(".TW", "")
                    bt_df = fm_api.taiwan_stock_daily(stock_id=fm_id, start_date=start_date_str, end_date=end_date_str)
                    if not bt_df.empty:
                        bt_df = bt_df.rename(columns={'date': 'Date', 'close': 'Close'})
                        bt_df['Date'] = pd.to_datetime(bt_df['Date'])
                        bt_data = bt_df.set_index('Date')
                
                if not bt_data.empty and len(bt_data) > 1:
                    # 🌟 核心防呆：抹除時區資訊避免 to_period 時異常
                    if bt_data.index.tz is not None:
                        bt_data.index = bt_data.index.tz_localize(None)
                        
                    bt_data['YearMonth'] = bt_data.index.to_period('M')
                    monthly_first_days = bt_data.groupby('YearMonth').first()
                    
                    total_months = len(monthly_first_days)
                    total_cost = total_months * t2_amount
                    total_shares = (t2_amount / monthly_first_days['Close']).sum()
                    final_value = total_shares * bt_data['Close'].iloc[-1]
                    
                    st.success(f"✅ 回測完成！ (數據來源: {actual_t2_source}) | 期間：{monthly_first_days.index[0]} 至 {monthly_first_days.index[-1]} (共 {total_months} 個月)")
                    res_c1, res_c2, res_c3 = st.columns(3)
                    with res_c1: st.metric("總投入成本", f"NT$ {total_cost:,.0f}")
                    with res_c2: st.metric("期末總價值", f"NT$ {final_value:,.0f}", f"{final_value - total_cost:+,.0f}")
                    with res_c3: st.metric("總報酬率 (ROI)", f"{((final_value - total_cost) / total_cost * 100):.2f}%")
                else:
                    st.error("❌ 獲取資料失敗，請確認該標的在指定日期已上市。")
            except Exception:
                st.error("資料處理發生錯誤。")

# ------------------------------------------
# 🧠 第三分頁：市場趨勢即時掃描
# ------------------------------------------
with tab3:
    with st.container(border=True):
        st.subheader("📺 市場趨勢即時掃描 (無限自訂 + 智能分流)")
        
        active_etf_list = [f"00{970+i}A.TW" for i in range(28)]
        pools = {
            "🇹🇼 0050 前十大權值股": ["2330.TW", "2454.TW", "2308.TW", "2317.TW", "3711.TW", "2383.TW", "2303.TW", "3037.TW", "2345.TW", "2891.TW"],
            "🇹🇼 熱門高股息 ETF": ["0056.TW", "00878.TW", "00713.TW", "00919.TW", "00929.TW", "00939.TW", "00940.TW"],
            "🇹🇼 半導體與科技主題 ETF": ["00891.TW", "00892.TW", "00881.TW", "00904.TW", "00927.TW", "0052.TW"],
            "🇹🇼 2026 全市場主動式 ETF (28檔)": active_etf_list,
            "🇺🇸 美股科技巨頭與半導體": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMD", "AMZN", "META", "AVGO"],
            "🌍 全球宏觀指標 (金/油/大盤)": ["GC=F", "CL=F", "^TWII", "^GSPC", "^TNX"],
            "✏️ 自訂掃描清單 (無限制)": [] 
        }
        
        name_dict = {
            "2330.TW": "台積電", "2454.TW": "聯發科", "2308.TW": "台達電", "2317.TW": "鴻海", 
            "3711.TW": "日月光投控", "2383.TW": "台光電", "2303.TW": "聯電", "3037.TW": "欣興", 
            "2345.TW": "智邦", "2891.TW": "中信金",
            "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00713.TW": "元大台灣高息低波", 
            "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息", "00939.TW": "統一台灣高息動能", "00940.TW": "元大台灣價值高息",
            "00891.TW": "中信關鍵半導體", "00892.TW": "富邦台灣半導體", "00881.TW": "國泰台灣5G+", 
            "00904.TW": "新光臺灣半導體30", "00927.TW": "群益半導體收益", "0052.TW": "富邦科技",
            "AAPL": "蘋果", "MSFT": "微軟", "NVDA": "輝達", "GOOGL": "Google", "TSLA": "特斯拉",
            "AMD": "超微", "AMZN": "亞馬遜", "META": "Meta", "AVGO": "博通",
            "GC=F": "黃金期貨 (美元/盎司)", "CL=F": "原油期貨 (WTI)", "^TWII": "台灣加權指數", "^GSPC": "標普500指數", "^TNX": "美國10年期公債殖利率"
        }

        holdings_dict = {
            "2330.TW": "無 (個股)", "2454.TW": "無 (個股)", "2308.TW": "無 (個股)", "2317.TW": "鴻海",
            "3711.TW": "無 (個股)", "2383.TW": "無 (個股)", "2303.TW": "無 (個股)", "3037.TW": "無 (個股)",
            "2345.TW": "無 (個股)", "2891.TW": "無 (個股)",
            "0056.TW": "聯發科, 鴻海, 廣達", "00878.TW": "華碩, 大聯大, 聯發科", "00713.TW": "統一, 台灣大, 遠傳", 
            "00919.TW": "長榮, 聯發科, 瑞昱", "00929.TW": "聯發科, 瑞昱, 聯詠", "00939.TW": "聯發科, 緯創, 大聯大", "00940.TW": "長榮, 聯電, 中美晶",
            "00891.TW": "聯發科, 台積電, 日月光", "00892.TW": "台積電, 聯發科, 聯電", "00881.TW": "台積電, 鴻海, 聯發科", 
            "00904.TW": "台積電, 聯發科, 聯詠", "00927.TW": "聯發科, 台積電, 瑞昱", "0052.TW": "台積電, 聯發科, 鴻海",
            "AAPL": "無 (個股)", "MSFT": "無 (個股)", "NVDA": "無 (個股)", "GOOGL": "無 (個股)", "TSLA": "無 (個股)",
            "AMD": "無 (個股)", "AMZN": "無 (個股)", "META": "無 (個股)", "AVGO": "博通",
            "GC=F": "無 (大宗商品)", "CL=F": "無 (大宗商品)", "^TWII": "無 (大盤指數)", "^GSPC": "無 (大盤指數)", "^TNX": "無 (債券殖利率)"
        }

        for t in active_etf_list:
            name_dict[t] = f"主動型 {t[:6]}"
            holdings_dict[t] = "機密 (經理人動態選股)"

        scan_c1, scan_c2, scan_c3 = st.columns([1.5, 1, 1.5])
        with scan_c1:
            scan_pool_name = st.selectbox("選擇實時運算模組", list(pools.keys()), key="t3_pool")
            
            if scan_pool_name == "✏️ 自訂掃描清單 (無限制)":
                custom_tickers_input = st.text_input("輸入標的代碼 (逗號分隔，台股建議加 .TW)", "2330.TW, GC=F, AAPL, NVDA", key="t3_custom")
                tickers_to_fetch = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()]
            else:
                tickers_to_fetch = pools[scan_pool_name]
                
        with scan_c2:
            scan_source = st.radio("底層 API 路由策略", ["自動 (台股FinMind/其他Yahoo)", "強制 Yahoo", "強制 FinMind"], key="t3_src")
            
            total_tickers = len(tickers_to_fetch)
            if total_tickers <= 5:
                count_options = ["顯示全部 (1~5檔)", 3]
            elif total_tickers <= 10:
                count_options = [f"顯示全部 ({total_tickers}檔)", 3, 5]
            else:
                count_options = [f"顯示全部 ({total_tickers}檔)", 3, 5, 10, 20]
                
            scan_count = st.selectbox("顯示結果數量", count_options, key="t3_cnt")
            
        with scan_c3:
            scan_strategy = st.selectbox("即時排序策略", ["依 今年來漲幅(%) 由高到低", "依 當前 RSI 由高到低 (動能強)", "依 當前 RSI 由低到高 (超跌區)"], key="t3_strat")
            
        t3_button = st.button("🚀 啟動全網大掃描", use_container_width=True, key="t3_btn")

    if t3_button:
        if not tickers_to_fetch:
            st.warning("請輸入至少一檔股票或指標代碼。")
            st.stop()
            
        progress_bar = st.progress(0, text="準備連線抓取資料...")
        scan_results = []
        
        current_year = datetime.date.today().year
        end_date_str = datetime.date.today().strftime("%Y-%m-%d")
        fetch_start_date = (datetime.date(current_year, 1, 1) - datetime.timedelta(days=45)).strftime("%Y-%m-%d")

        for i, ticker in enumerate(tickers_to_fetch):
            progress_bar.progress((i + 1) / len(tickers_to_fetch), text=f"正在分析 {name_dict.get(ticker, ticker)}...")
            
            is_tw_stock = ticker.endswith(".TW") or ticker.endswith(".TWO") or (ticker.isdigit() and len(ticker) >= 4)
            use_fm = False
            
            if scan_source == "自動 (台股FinMind/其他Yahoo)": use_fm = is_tw_stock
            elif scan_source == "強制 FinMind": use_fm = True
                
            try:
                stock_data = pd.DataFrame()
                
                if use_fm:
                    fm_id = ticker.replace(".TW", "").replace(".TWO", "")
                    fm_df = fm_api.taiwan_stock_daily(stock_id=fm_id, start_date=fetch_start_date, end_date=end_date_str)
                    if not fm_df.empty and len(fm_df) > 15:
                        fm_df = fm_df.rename(columns={'date': 'Date', 'close': 'Close'})
                        fm_df['Date'] = pd.to_datetime(fm_df['Date'])
                        stock_data = fm_df.set_index('Date')
                else:
                    stock_data = yf.Ticker(ticker).history(start=fetch_start_date, end=end_date_str, auto_adjust=True)
                
                if not stock_data.empty and len(stock_data) > 15:
                    if stock_data.index.tz is not None: stock_data.index = stock_data.index.tz_localize(None)
                        
                    current_price = stock_data['Close'].iloc[-1]
                    ytd_data = stock_data[stock_data.index.year == current_year]
                    ytd_return = ((current_price - ytd_data['Close'].iloc[0]) / ytd_data['Close'].iloc[0]) * 100 if not ytd_data.empty else 0.0

                    delta = stock_data['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.ewm(com=13, adjust=False).mean()
                    avg_loss = loss.ewm(com=13, adjust=False).mean()
                    rs = avg_gain / avg_loss
                    current_rsi = (np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs))))[-1]
                    
                    if is_tw_stock: flag = "🇹🇼 "
                    elif ticker in ["GC=F", "CL=F", "^GSPC", "^TNX"]: flag = "🌍 "
                    else: flag = "🇺🇸 "
                    
                    scan_results.append({
                        "代號": ticker, "標的名稱": flag + name_dict.get(ticker, ticker), 
                        "前三大持股 (透視)": holdings_dict.get(ticker, "無資料"),
                        "最新收盤價": round(current_price, 2), "今年來漲幅(%)": round(ytd_return, 2),
                        "當前 RSI": round(current_rsi, 1), "來源": "FinMind" if use_fm else "Yahoo"
                    })
            except Exception:
                pass 
        
        progress_bar.empty()
        
        if len(scan_results) > 0:
            df_results = pd.DataFrame(scan_results)
            
            if "今年來漲幅" in scan_strategy: df_results = df_results.sort_values(by="今年來漲幅(%)", ascending=False)
            elif "RSI 由高到低" in scan_strategy: df_results = df_results.sort_values(by="當前 RSI", ascending=False)
            elif "RSI 由低到高" in scan_strategy: df_results = df_results.sort_values(by="當前 RSI", ascending=True)
                
            if not scan_count.startswith("顯示全部"): 
                df_results = df_results.head(int(scan_count))
                
            display_df = df_results.reset_index(drop=True)
            display_df.index = display_df.index + 1
            
            st.success(f"✅ 掃描完成！共成功抓取 {len(scan_results)} 筆即時市場數據。")
            st.dataframe(display_df, use_container_width=True, column_config={
                "最新收盤價": st.column_config.NumberColumn("收盤報價", format="%.2f"),
                "今年來漲幅(%)": st.column_config.NumberColumn("今年漲幅(%)", format="%.2f %%"),
                "當前 RSI": st.column_config.ProgressColumn("短期動能 (RSI)", format="%.1f", min_value=0, max_value=100)
            })
        else:
            st.error("❌ 獲取資料失敗。可能是 API 連線限制或輸入的代碼無效。")

# ============================================================
# Version 5 AI 核心函式
# ============================================================
def validate_stock_id(stock_id: str) -> bool:
    """驗證台股或 ETF 代碼。"""
    stock_id = stock_id.strip().upper()
    return stock_id.isalnum() and 4 <= len(stock_id) <= 6


def get_date_range(years: int = 5):
    """取得模型訓練所需的歷史日期範圍。"""
    today = date.today()
    start = today - relativedelta(years=years)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def normalize_finmind_columns(df: pd.DataFrame) -> pd.DataFrame:
    """將 FinMind 欄位整理為統一 OHLCV 格式。"""
    if df.empty:
        return df

    rename_map = {
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
        "volume": "Volume"
    }
    df = df.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FinMind 回傳資料缺少欄位：{missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ============================================================
# 2. 股票資料下載
# ============================================================

def get_finmind_data(stock_id: str, years: int = 5) -> pd.DataFrame:
    start_date, end_date = get_date_range(years)

    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return normalize_finmind_columns(df)


def get_yfinance_data(stock_id: str, years: int = 5) -> pd.DataFrame:
    start_date, end_date = get_date_range(years)
    ticker = f"{stock_id}.TW"

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance 某些版本會回傳 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in required if c in df.columns]].copy()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def get_stock_data(
    stock_id: str,
    data_source: str = "finmind",
    years: int = 5
) -> pd.DataFrame:
    """依資料來源下載股票資料。"""
    stock_id = stock_id.strip().upper()

    try:
        if data_source == "yfinance":
            df = get_yfinance_data(stock_id, years)
        else:
            df = get_finmind_data(stock_id, years)

        if df.empty:
            print("⚠️ 查無資料，請確認股票代碼或資料來源。")
            return df

        print(
            f"✅ 取得 {len(df)} 筆資料："
            f"{df.index.min().date()} ~ {df.index.max().date()}"
        )
        return df

    except Exception as e:
        print(f"❌ 股票資料下載失敗：{e}")
        return pd.DataFrame()


# ============================================================
# 3. 新聞搜尋、關鍵字權重與 FinBERT 情緒分析
# ============================================================

# 真正的新聞搜尋關鍵字：程式會在取得股票名稱後自動組合
# 例如：AI 台積電、營收 台積電、法說會 台積電。
NEWS_QUERY_TERMS = [
    "AI",
    "營收",
    "財報",
    "獲利",
    "法說會",
    "半導體",
    "產能",
    "訂單",
    "展望",
    "投資",
    "產品",
    "需求",
    "景氣",
    "政策",
    "利率"
]

NEWS_RESULTS_PER_QUERY = 10
NEWS_MAX_DISPLAY = 30
NEWS_RECENCY_DECAY = 0.05


def get_stock_name(stock_id: str) -> str:
    """取得股票中文名稱；若查不到則退回股票代碼。"""
    try:
        info = api.taiwan_stock_info()
        if info is not None and not info.empty:
            id_col = next(
                (c for c in ["stock_id", "StockID"] if c in info.columns),
                None
            )
            name_col = next(
                (c for c in ["stock_name", "StockName", "name"] if c in info.columns),
                None
            )
            if id_col and name_col:
                matched = info[info[id_col].astype(str).str.upper() == stock_id.upper()]
                if not matched.empty:
                    return str(matched.iloc[0][name_col]).strip()
    except Exception as e:
        print(f"⚠️ 無法由 FinMind 取得股票名稱：{e}")

    # 常見台股 fallback，避免 API 偶爾查不到名稱時無法建立搜尋詞。
    fallback_names = {
        "2330": "台積電",
        "2317": "鴻海",
        "2454": "聯發科",
        "2308": "台達電",
        "2382": "廣達",
        "2303": "聯電",
        "2881": "富邦金",
        "2882": "國泰金",
        "0050": "元大台灣50",
        "0056": "元大高股息",
        "00878": "國泰永續高股息"
    }
    return fallback_names.get(stock_id.upper(), stock_id.upper())


def build_news_queries(stock_name: str) -> list[str]:
    """建立「新聞關鍵字 + 股票名稱」的真正搜尋查詢。"""
    return [f"{term} {stock_name}" for term in NEWS_QUERY_TERMS]


def search_google_news(query: str, max_results: int = NEWS_RESULTS_PER_QUERY) -> list[dict]:
    """使用 Google News RSS 搜尋真正的新聞結果，不需要 API Key。"""
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=12) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    items = []

    for item in root.findall("./channel/item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()

        if not title:
            continue

        items.append({
            "Query": query,
            "Title": title,
            "Link": link,
            "PubDate": pub_date,
            "Description": description
        })

    return items


def label_to_score(label: str) -> float:
    """正向=+1、負向=-1、中立=0。"""
    label = str(label).lower()
    positive_words = ["positive", "pos", "bullish", "利多", "正向"]
    negative_words = ["negative", "neg", "bearish", "利空", "負向"]

    if any(word in label for word in positive_words):
        return 1.0
    if any(word in label for word in negative_words):
        return -1.0
    return 0.0


def calculate_news_weight(
    news_date: pd.Timestamp,
    latest_date: pd.Timestamp,
    confidence: float
) -> float:
    """新聞權重 = 新鮮度 × FinBERT 信心度。"""
    days_old = max((latest_date - news_date).days, 0)
    recency_weight = np.exp(-NEWS_RECENCY_DECAY * days_old)
    return float(recency_weight * confidence)


def get_news_sentiment(
    stock_id: str,
    stock_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    股票名稱 → 自動建立「關鍵字 + 股票名稱」→ Google News RSS 搜尋
    → 去除重複新聞 → FinBERT → 新聞權重 → 每日情緒。
    """
    result = pd.DataFrame(index=stock_index)
    result["sentiment_score_raw"] = 0.0
    result["sentiment_score"] = 0.0

    stock_name = get_stock_name(stock_id)
    queries = build_news_queries(stock_name)

    print("\n" + "=" * 90)
    print("📰 真正的新聞搜尋設定")
    print("=" * 90)
    print(f"股票代碼：{stock_id}")
    print(f"股票名稱：{stock_name}")
    print("本次實際搜尋關鍵字：")
    for i, query in enumerate(queries, 1):
        print(f"  {i:2d}. \"{query}\"")
    print("=" * 90)

    all_news = []
    for query in queries:
        try:
            found = search_google_news(query)
            all_news.extend(found)
            print(f"🔎 {query:<24} → 找到 {len(found):2d} 則")
        except Exception as e:
            print(f"⚠️ {query:<24} → 搜尋失敗：{e}")

    if not all_news:
        print("⚠️ 所有新聞搜尋都失敗，情緒特徵以 0 補值。")
        return result

    news = pd.DataFrame(all_news)
    news["Date"] = pd.to_datetime(news["PubDate"], errors="coerce", utc=True).dt.tz_convert(None)
    news = news.dropna(subset=["Date"]).copy()
    news["Date"] = news["Date"].dt.normalize()
    news["News_Text"] = news["Title"].astype(str).str.strip()

    # 同一新聞可能被多個搜尋詞找到；保留第一筆並記錄所有命中的查詢。
    news["Dedup_Key"] = news["Title"].str.lower().str.replace(r"\\s+", " ", regex=True)
    grouped = []
    for key, group in news.groupby("Dedup_Key", sort=False):
        row = group.iloc[0].copy()
        row["Matched_Queries"] = " | ".join(dict.fromkeys(group["Query"].tolist()))
        row["Keyword_Count"] = group["Query"].nunique()
        grouped.append(row)

    news = pd.DataFrame(grouped)

    # 只使用模型歷史資料時間範圍內的新聞，避免未來日期污染訓練資料。
    min_date = stock_index.min().normalize()
    max_date = stock_index.max().normalize()
    news = news[(news["Date"] >= min_date) & (news["Date"] <= max_date)].copy()

    if news.empty:
        print("⚠️ 搜尋到的新聞不在股票歷史資料日期範圍內。")
        return result

    latest_date = news["Date"].max()
    print(f"\n原始搜尋結果：{len(all_news)} 則")
    print(f"去除重複後：{len(news)} 則")
    print(f"分析期間：{min_date.date()} ~ {max_date.date()}")
    print("\n🤖 正在進行 FinBERT 情緒分析...")

    rows = []
    for _, row in news.iterrows():
        text = row["News_Text"][:500]
        if not text:
            continue

        try:
            pred = ensure_sentiment_classifier()(text)[0]
            label = pred.get("label", "neutral")
            confidence = float(pred.get("score", 0.0))
            sentiment_base = label_to_score(label)
            sentiment_value = sentiment_base * confidence
            raw_weight = calculate_news_weight(row["Date"], latest_date, confidence)

            rows.append({
                "Date": row["Date"],
                "News": text,
                "Link": row["Link"],
                "Matched_Queries": row["Matched_Queries"],
                "Keyword_Count": int(row["Keyword_Count"]),
                "Label": label,
                "Confidence": confidence,
                "Sentiment": sentiment_value,
                "Raw_Weight": raw_weight
            })
        except Exception:
            continue

    if not rows:
        print("⚠️ FinBERT 沒有成功分析任何新聞。")
        return result

    detail = pd.DataFrame(rows)
    total_weight = detail["Raw_Weight"].sum()
    detail["Weight"] = (
        detail["Raw_Weight"] / total_weight
        if total_weight > 0
        else 1.0 / len(detail)
    )
    detail["Weight_Pct"] = detail["Weight"] * 100
    detail["Weighted_Sentiment"] = detail["Sentiment"] * detail["Weight"]

    # --------------------------------------------------------
    # 每則新聞
    # --------------------------------------------------------
    print("\n" + "=" * 110)
    print("📋 新聞權重明細")
    print("=" * 110)
    display_detail = detail.sort_values(["Date", "Weight"], ascending=[False, False]).head(NEWS_MAX_DISPLAY)

    for _, item in display_detail.iterrows():
        print(
            f"{item['Date'].strftime('%Y-%m-%d')} | "
            f"{item['Label']:<10} | "
            f"信心={item['Confidence']:.3f} | "
            f"權重={item['Weight_Pct']:.2f}% | "
            f"命中關鍵字={item['Keyword_Count']}"
        )
        print(f"    搜尋詞：{item['Matched_Queries']}")
        print(f"    {item['News']}")

    # --------------------------------------------------------
    # 各搜尋關鍵字占比
    # 一則新聞若同時被多個 query 找到，權重平均分配給命中的 query。
    # --------------------------------------------------------
    keyword_rows = []
    for _, item in detail.iterrows():
        matched = [x.strip() for x in str(item["Matched_Queries"]).split("|") if x.strip()]
        if not matched:
            continue
        share = item["Weight"] / len(matched)
        for query in matched:
            keyword_rows.append({"Query": query, "Weight": share})

    keyword_df = pd.DataFrame(keyword_rows)
    if not keyword_df.empty:
        keyword_summary = keyword_df.groupby("Query")["Weight"].sum().sort_values(ascending=False)
        print("\n" + "=" * 90)
        print("📊 各新聞搜尋關鍵字的權重占比")
        print("=" * 90)
        for query, weight in keyword_summary.items():
            count = sum(query in str(x) for x in detail["Matched_Queries"])
            print(f"{query:<24} 新聞命中={count:3d} | 權重={weight * 100:6.2f}%")
        print(f"{'總計':<24}                 | 權重={keyword_summary.sum() * 100:6.2f}%")
        print("=" * 90)

    # --------------------------------------------------------
    # 情緒權重分布
    # --------------------------------------------------------
    positive_weight = detail.loc[detail["Sentiment"] > 0, "Weight"].sum()
    negative_weight = detail.loc[detail["Sentiment"] < 0, "Weight"].sum()
    neutral_weight = detail.loc[detail["Sentiment"] == 0, "Weight"].sum()

    print("\n" + "=" * 72)
    print("📊 新聞情緒權重分布")
    print("=" * 72)
    print(f"正向新聞權重：{positive_weight * 100:6.2f}%")
    print(f"中立新聞權重：{neutral_weight * 100:6.2f}%")
    print(f"負向新聞權重：{negative_weight * 100:6.2f}%")
    print(f"總權重：        {detail['Weight'].sum() * 100:6.2f}%")
    print("=" * 72)

    # 每日情緒：讓新聞真正對應到歷史交易日。
    daily = detail.groupby("Date").apply(
        lambda g: np.average(g["Sentiment"], weights=g["Weight"])
        if g["Weight"].sum() > 0 else 0.0,
        include_groups=False
    )

    result["sentiment_score_raw"] = daily.reindex(result.index).fillna(0.0)
    result["sentiment_score"] = result["sentiment_score_raw"].rolling(20, min_periods=1).mean()

    print("✅ 真正新聞搜尋 + 關鍵字權重 + FinBERT 情緒分析完成。")
    return result


# ============================================================
# 4. 技術指標與特徵工程
# ============================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # 波動率
    df["Volatility20"] = (
        df["Return"].rolling(20).std()
    )

    # ATR(14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()

    # 成交量變化
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    return df


def prepare_feature_data(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """合併股票、技術指標與情緒資料。"""
    df = add_technical_indicators(stock_df)

    df = df.join(
        sentiment_df[
            ["sentiment_score_raw", "sentiment_score"]
        ],
        how="left"
    )

    df["sentiment_score_raw"] = (
        df["sentiment_score_raw"].fillna(0.0)
    )
    df["sentiment_score"] = (
        df["sentiment_score"].fillna(0.0)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# 5. KMeans 市場狀態辨識
# ============================================================

def build_kmeans_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    每一列代表截至當日的市場狀態。
    KMeans 使用趨勢、波動與動能特徵分群。
    """
    x = pd.DataFrame(index=df.index)

    x["Trend20"] = df["Close"].pct_change(20)
    x["Trend60"] = df["Close"].pct_change(60)
    x["Volatility20"] = df["Return"].rolling(20).std()
    x["RSI14"] = df["RSI14"]
    x["MACD"] = df["MACD"] / df["Close"]
    x["Volume_Ratio"] = (
        df["Volume"] /
        df["Volume_MA20"].replace(0, np.nan)
    )

    return x.replace([np.inf, -np.inf], np.nan).dropna()


def get_market_state(
    df: pd.DataFrame,
    n_clusters: int = KMEANS_CLUSTERS
):
    """
    KMeans 分群後，依群中心的趨勢特徵命名：
    上升型、下降型、震盪型。
    """
    cluster_features = build_kmeans_features(df)

    if len(cluster_features) < 100:
        raise ValueError(
            "KMeans 資料不足，建議至少取得 1 年以上資料。"
        )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cluster_features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=20
    )

    labels = model.fit_predict(x_scaled)

    cluster_features = cluster_features.copy()
    cluster_features["Cluster"] = labels

    # 以 Trend20 + Trend60 判斷群組趨勢
    summary = cluster_features.groupby("Cluster").agg(
        Trend20=("Trend20", "mean"),
        Trend60=("Trend60", "mean"),
        Volatility20=("Volatility20", "mean")
    )

    trend_score = summary["Trend20"] + summary["Trend60"]

    up_cluster = trend_score.idxmax()
    down_cluster = trend_score.idxmin()

    state_map = {}
    for cluster_id in summary.index:
        if cluster_id == up_cluster:
            state_map[cluster_id] = "上升型"
        elif cluster_id == down_cluster:
            state_map[cluster_id] = "下降型"
        else:
            state_map[cluster_id] = "震盪型"

    # 將每日 Cluster 加回主資料
    cluster_series = pd.Series(
        labels,
        index=cluster_features.index,
        name="Market_Cluster"
    )

    state_series = cluster_series.map(state_map)
    state_series.name = "Market_State"

    latest_cluster = int(cluster_series.iloc[-1])
    latest_state = state_map[latest_cluster]

    return (
        cluster_series,
        state_series,
        latest_cluster,
        latest_state,
        summary,
        model,
        scaler
    )


# ============================================================
# 6. TCN Dataset
# ============================================================

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "MA5",
    "MA20",
    "MA60",
    "RSI14",
    "MACD",
    "MACD_Signal",
    "Volatility20",
    "ATR14",
    "Volume_Change",
    "sentiment_score"
]

#
TARGET_COLUMNS = ["Open_Return", "High_Return", "Low_Return", "Close_Return"]

#
def create_tcn_dataset(
    df: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS,
    horizon: int = FORECAST_DAYS
):
    """
    X：過去 60 天特徵
    y：未來 22 天 OHLC 報酬率
    """

    clean = df.copy()

    # ==========================================
    # 建立 OHLC 相對前一交易日 Close 的報酬率
    # ==========================================
    previous_close = clean["Close"].shift(1)

    clean["Open_Return"] = clean["Open"] / previous_close - 1
    clean["High_Return"] = clean["High"] / previous_close - 1
    clean["Low_Return"] = clean["Low"] / previous_close - 1
    clean["Close_Return"] = clean["Close"] / previous_close - 1

    clean = clean.dropna(
        subset=FEATURE_COLUMNS + TARGET_COLUMNS
    ).copy()

    if len(clean) < lookback + horizon + 30:
        raise ValueError(
            f"資料不足，至少需要 {lookback + horizon + 30} 筆有效交易日。"
        )

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_values = feature_scaler.fit_transform(
        clean[FEATURE_COLUMNS]
    )

    # ==========================================
    # 模型現在學習的是「報酬率」
    # ==========================================
    target_values = target_scaler.fit_transform(
        clean[TARGET_COLUMNS]
    )

    x_list = []
    y_list = []

    for end_idx in range(
        lookback,
        len(clean) - horizon + 1
    ):
        start_idx = end_idx - lookback

        x_list.append(
            feature_values[start_idx:end_idx]
        )

        future_return = target_values[
            end_idx:end_idx + horizon
        ]

        y_list.append(
            future_return.reshape(-1)
        )

    X = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return (
        X,
        y,
        clean,
        feature_scaler,
        target_scaler
    )


def tcn_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float = 0.15
):
    """TCN 的 causal dilated residual block。"""
    residual = x

    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(x)
    y = BatchNormalization()(y)
    y = Dropout(dropout_rate)(y)

    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(y)
    y = BatchNormalization()(y)
    y = Dropout(dropout_rate)(y)

    if residual.shape[-1] != filters:
        residual = Conv1D(
            filters,
            kernel_size=1,
            padding="same"
        )(residual)

    return Add()([residual, y])


def build_tcn_model(
    lookback: int,
    n_features: int,
    horizon: int
):
    """
    TCN：
    60 天輸入 → 多層 causal dilated Conv1D → 22 天 OHLC。
    """
    inputs = tf.keras.Input(
        shape=(lookback, n_features)
    )

    x = Conv1D(
        64,
        kernel_size=3,
        padding="causal",
        activation="relu"
    )(inputs)

    # Dilated convolution 讓模型能看到不同時間尺度
    for dilation in [1, 2, 4, 8, 16]:
        x = tcn_residual_block(
            x,
            filters=64,
            kernel_size=3,
            dilation_rate=dilation,
            dropout_rate=0.15
        )

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.20)(x)
    x = Dense(64, activation="relu")(x)

    outputs = Dense(
        horizon * 4,
        name="future_returns"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_tcn_model(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int
):
    """依時間順序切分 TCN 訓練/驗證資料。"""
    split = int(len(X) * 0.8)

    if split < 30 or len(X) - split < 5:
        raise ValueError("TCN 訓練/驗證樣本不足。")

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_tcn_model(
        lookback=lookback,
        n_features=X.shape[2],
        horizon=horizon
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    print(
        f"🤖 開始訓練 TCN："
        f"訓練 {len(X_train)} 筆，"
        f"驗證 {len(X_val)} 筆..."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history, (X_val, y_val)


def make_latest_input(
    clean_df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    lookback: int
) -> np.ndarray:
    latest = clean_df[FEATURE_COLUMNS].iloc[-lookback:]

    if len(latest) != lookback:
        raise ValueError("最新資料不足 60 個交易日。")

    x = feature_scaler.transform(latest)
    return x.reshape(1, lookback, len(FEATURE_COLUMNS))


def repair_ohlc(
    ohlc: np.ndarray,
    last_close: float
) -> np.ndarray:
    """修正 OHLC 邏輯，避免產生不合理 K 線。"""
    fixed = np.asarray(ohlc, dtype=float).copy()

    for i in range(len(fixed)):
        open_p, high_p, low_p, close_p = fixed[i]

        if i == 0:
            open_p = 0.7 * last_close + 0.3 * open_p
        else:
            open_p = 0.7 * fixed[i - 1, 3] + 0.3 * open_p

        body_high = max(open_p, close_p)
        body_low = min(open_p, close_p)

        high_p = max(high_p, body_high)
        low_p = min(low_p, body_low)

        open_p = max(open_p, 0.01)
        high_p = max(high_p, 0.01)
        low_p = max(low_p, 0.01)
        close_p = max(close_p, 0.01)

        fixed[i] = [
            open_p,
            high_p,
            low_p,
            close_p
        ]

    return fixed


def predict_future_ohlc(
    model,
    clean_df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    lookback: int = LOOKBACK_DAYS,
    horizon: int = FORECAST_DAYS
) -> pd.DataFrame:

    x_latest = make_latest_input(
        clean_df,
        feature_scaler,
        lookback
    )

    # ==========================================
    # TCN 預測未來 22 天報酬率
    # ==========================================
    pred_scaled = model.predict(
        x_latest,
        verbose=0
    )[0]

    pred_scaled = pred_scaled.reshape(
        horizon, 4
    )

    # 還原成實際報酬率
    pred_returns = target_scaler.inverse_transform(
        pred_scaled
    )

    # ==========================================
    # 限制每日報酬率在 -10% ~ +10%
    # ==========================================
    pred_returns = np.clip(
        pred_returns,
        -0.10,
        0.10
    )

    last_close = float(
        clean_df["Close"].iloc[-1]
    )

    # ==========================================
    # 將報酬率還原成 OHLC 價格
    # ==========================================
    forecast_ohlc = []

    previous_close = last_close

    for i in range(horizon):

        open_return = pred_returns[i, 0]
        high_return = pred_returns[i, 1]
        low_return = pred_returns[i, 2]
        close_return = pred_returns[i, 3]

        open_p = previous_close * (1 + open_return)
        high_p = previous_close * (1 + high_return)
        low_p = previous_close * (1 + low_return)
        close_p = previous_close * (1 + close_return)

        # 確保 K 線邏輯合理
        high_p = max(
            high_p,
            open_p,
            close_p
        )

        low_p = min(
            low_p,
            open_p,
            close_p
        )

        forecast_ohlc.append([
            open_p,
            high_p,
            low_p,
            close_p
        ])

        # 下一天以上一天預測 Close 為基準
        previous_close = close_p

    forecast_ohlc = np.asarray(
        forecast_ohlc,
        dtype=float
    )

    future_dates = pd.bdate_range(
        start=clean_df.index[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast = pd.DataFrame(
        forecast_ohlc,
        index=future_dates,
        columns=["Open", "High", "Low", "Close"]
    )

    forecast["Volume"] = 0.0

    # ==========================================
    # 每日 Close 報酬率
    # ==========================================
    previous_close_series = pd.concat([
        pd.Series(
            [last_close],
            index=[clean_df.index[-1]]
        ),
        forecast["Close"]
    ])

    forecast["Return"] = (
        forecast["Close"].to_numpy()
        / previous_close_series.iloc[:-1].to_numpy()
        - 1
    )

    # 再保險一次，確保顯示的每日報酬率一定在 ±10%
    forecast["Return"] = np.clip(
        forecast["Return"],
        -0.10,
        0.10
    )

    return forecast


# ============================================================
# 8. 模型評估
# ============================================================

def evaluate_tcn(
    model,
    validation_data,
    target_scaler: MinMaxScaler,
    horizon: int = FORECAST_DAYS
):
    X_val, y_val = validation_data

    pred = model.predict(
        X_val,
        verbose=0
    )

    pred = pred.reshape(
        -1, horizon, 4
    )
    actual = y_val.reshape(
        -1, horizon, 4
    )

    pred_flat = pred.reshape(-1, 4)
    actual_flat = actual.reshape(-1, 4)

    pred_real = target_scaler.inverse_transform(
        pred_flat
    )
    actual_real = target_scaler.inverse_transform(
        actual_flat
    )

    metrics = {}

    for i, name in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(
            actual_real[:, i],
            pred_real[:, i]
        )
        rmse = np.sqrt(
            mean_squared_error(
                actual_real[:, i],
                pred_real[:, i]
            )
        )

        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse)
        }

    return metrics


# ============================================================
# 9. K 線圖
# ============================================================

def plot_history_and_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    stock_id: str,
    market_state: str,
    history_days: int = 22
):
    """
    歷史 K 線：
      紅漲、綠跌
    預測 K 線：
      藍漲、橘跌
    """

    historical = history_df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].tail(history_days).copy()

    future = forecast_df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    # 分別建立兩種顏色的 style
    history_colors = mpf.make_marketcolors(
        up="red",
        down="green",
        edge="inherit",
        wick="inherit",
        volume="inherit"
    )

    history_style = mpf.make_mpf_style(
        marketcolors=history_colors,
        rc={
            "font.family": [
                "Microsoft JhengHei",
                "Arial"
            ]
        }
    )

    fig, axes = mpf.plot(
        historical,
        type="candle",
        volume=True,
        style=history_style,
        title=(
            f"{stock_id} 歷史 K 線與未來一個月 AI 預測\n"
            f"KMeans 目前市場狀態：{market_state}"
        ),
        ylabel="價格",
        ylabel_lower="成交量",
        returnfig=True,
        figsize=(15, 9)
    )

    ax_price = axes[0]

    # 預測 K 線用 matplotlib 手動畫，
    # 可獨立使用藍漲、橘跌。
    x_start = len(historical)

    for i, (_, row) in enumerate(future.iterrows()):
        x = x_start + i

        open_p = row["Open"]
        high_p = row["High"]
        low_p = row["Low"]
        close_p = row["Close"]

        up = close_p >= open_p
        color = "blue" if up else "orange"

        # 上下影線
        ax_price.vlines(
            x,
            low_p,
            high_p,
            color=color,
            linewidth=1.2,
            zorder=3
        )

        # 實體
        body_low = min(open_p, close_p)
        body_height = abs(close_p - open_p)

        if body_height < 1e-8:
            body_height = max(
                high_p - low_p,
                0.01
            ) * 0.03

        rect = plt.Rectangle(
            (x - 0.30, body_low),
            0.60,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
            zorder=4
        )

        ax_price.add_patch(rect)

    # 分隔歷史與預測
    ax_price.axvline(
        x=x_start - 0.5,
        color="gray",
        linestyle="--",
        linewidth=1.2
    )

    ax_price.text(
        x_start - 1,
        ax_price.get_ylim()[1],
        "歷史",
        ha="right",
        va="bottom"
    )

    ax_price.text(
        x_start,
        ax_price.get_ylim()[1],
        "AI 預測",
        ha="left",
        va="bottom",
        color="blue"
    )

    # 擴充 X 軸
    ax_price.set_xlim(
        -1,
        x_start + len(future)
    )

    plt.show()


# ============================================================
# 10. 輸出報表
# ============================================================

def print_market_report(
    stock_id: str,
    latest_cluster: int,
    market_state: str,
    cluster_summary: pd.DataFrame
):
    icon = {
        "上升型": "🟢",
        "下降型": "🔴",
        "震盪型": "🟡"
    }.get(market_state, "⚪")

    print("\n" + "=" * 68)
    print(f"📊 {stock_id} KMeans 市場狀態分析")
    print("=" * 68)
    print(
        f"目前 Cluster：{latest_cluster}"
    )
    print(
        f"目前市場狀態：{icon} {market_state}"
    )
    print("\n各 Cluster 平均特徵：")
    print(
        cluster_summary.round(4).to_string()
    )
    print("=" * 68)


def print_forecast_table(
    forecast_df: pd.DataFrame
):
    output = forecast_df.copy()

    output.insert(
        0,
        "Date",
        output.index.strftime("%Y-%m-%d")
    )

    output["Return"] = (
        output["Return"] * 100
    )

    output = output[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Return"
        ]
    ]

    print("\n" + "=" * 88)
    print("🔮 未來一個月預測 OHLC 與每日報酬率")
    print("=" * 88)

    print(
        output.to_string(
            index=False,
            formatters={
                "Open": "{:.2f}".format,
                "High": "{:.2f}".format,
                "Low": "{:.2f}".format,
                "Close": "{:.2f}".format,
                "Return": "{:+.2f}%".format
            }
        )
    )

    print("=" * 88)


def print_evaluation(metrics: dict):
    print("\n📏 TCN 驗證集評估")
    print("-" * 48)

    for name, values in metrics.items():
        print(
            f"{name:<5} "
            f"MAE={values['MAE']:.4f} | "
            f"RMSE={values['RMSE']:.4f}"
        )


# ============================================================
# 11. 主流程
# ============================================================

def run_prediction(
    stock_id: str,
    data_source: str
):
    print("\n" + "=" * 72)
    print(
        f"🚀 開始分析 {stock_id} "
        f"（資料來源：{data_source}）"
    )
    print("=" * 72)

    # 1. 股票資料
    stock_df = get_stock_data(
        stock_id,
        data_source=data_source,
        years=5
    )

    if stock_df.empty:
        return

    # 2. 新聞搜尋 + 情緒
    # 程式會自動建立「關鍵字 + 股票名稱」，不再要求使用者輸入新聞關鍵字。
    sentiment_df = get_news_sentiment(
        stock_id,
        stock_df.index
    )

    # 3. 技術指標與特徵
    feature_df = prepare_feature_data(
        stock_df,
        sentiment_df
    )

    # 4. KMeans
    try:
        (
            cluster_series,
            state_series,
            latest_cluster,
            latest_state,
            cluster_summary,
            _,
            _
        ) = get_market_state(feature_df)

        feature_df["Market_Cluster"] = (
            cluster_series.reindex(feature_df.index)
        )

        feature_df["Market_State"] = (
            state_series.reindex(feature_df.index)
        )

        print_market_report(
            stock_id,
            latest_cluster,
            latest_state,
            cluster_summary
        )

    except Exception as e:
        print(f"⚠️ KMeans 市場狀態分析失敗：{e}")
        latest_cluster = -1
        latest_state = "未知"

    # 5. 建立 TCN Dataset
    try:
        (
            X,
            y,
            clean_df,
            feature_scaler,
            target_scaler
        ) = create_tcn_dataset(
            feature_df,
            lookback=LOOKBACK_DAYS,
            horizon=FORECAST_DAYS
        )

        print(
            f"\n📦 TCN Dataset："
            f"X={X.shape}，y={y.shape}"
        )

    except Exception as e:
        print(f"❌ 建立 TCN Dataset 失敗：{e}")
        return

    # 6. 訓練 TCN
    try:
        (
            model,
            history,
            validation_data
        ) = train_tcn_model(
            X,
            y,
            lookback=LOOKBACK_DAYS,
            horizon=FORECAST_DAYS
        )

    except Exception as e:
        print(f"❌ TCN 訓練失敗：{e}")
        return

    # 7. 評估
    try:
        metrics = evaluate_tcn(
            model,
            validation_data,
            target_scaler,
            horizon=FORECAST_DAYS
        )
        print_evaluation(metrics)

    except Exception as e:
        print(f"⚠️ 模型評估失敗：{e}")

    # 8. 預測未來 OHLC
    try:
        forecast_df = predict_future_ohlc(
            model,
            clean_df,
            feature_scaler,
            target_scaler,
            lookback=LOOKBACK_DAYS,
            horizon=FORECAST_DAYS
        )

    except Exception as e:
        print(f"❌ 未來 OHLC 預測失敗：{e}")
        return

    # 9. 顯示表格
    print_forecast_table(
        forecast_df
    )

    # 10. 畫圖
    plot_history_and_forecast(
        history_df=stock_df,
        forecast_df=forecast_df,
        stock_id=stock_id,
        market_state=latest_state,
        history_days=22
    )

    print("\n✅ Version 5 執行完成。")



# ==========================================
# 🤖 第四分頁：Version 5 AI 多模態月度預測
# ==========================================
with tab4:
    with st.container(border=True):
        st.subheader("🤖 AI 多模態月度預測")
        st.caption("整合 Version 5：KMeans 市場狀態 + FinBERT 新聞情緒 + TCN 多步 OHLC 預測")

        ai_c1, ai_c2, ai_c3 = st.columns([1.5, 1.2, 1.3])
        with ai_c1:
            ai_target = st.text_input("輸入台股 / ETF 代碼：", "0050", key="ai_target")
            ai_target = ai_target.strip().upper().replace(".TW", "").replace(".TWO", "")
        with ai_c2:
            ai_source_label = st.radio("模型資料來源：", ("FinMind", "Yahoo Finance"), horizontal=True, key="ai_source")
            ai_source = "finmind" if ai_source_label == "FinMind" else "yfinance"
        with ai_c3:
            ai_years = st.selectbox("模型歷史資料：", [3, 5, 7, 10], index=1, format_func=lambda x: f"近 {x} 年", key="ai_years")

        # st.markdown("**🧠 Version 5 模型設定**")
        # model_c1, model_c2, model_c3, model_c4 = st.columns(4)
        # with model_c1: st.metric("輸入窗口", f"{LOOKBACK_DAYS} 交易日")
        # with model_c2: st.metric("預測區間", f"{FORECAST_DAYS} 交易日")
        # with model_c3: st.metric("KMeans", f"{KMEANS_CLUSTERS} 狀態")
        # with model_c4: st.metric("預測目標", "OHLC")

        ai_button = st.button("🚀 執行 Version 5 AI 預測", use_container_width=True, key="ai_btn")

    if ai_button:
        if not validate_stock_id(ai_target):
            st.error("❌ 股票 / ETF 代碼格式錯誤，請輸入 4～6 位英數代碼，例如 0050、2330。")
            st.stop()
        try:
            with st.spinner(f"正在建立 {ai_target} 的 AI 多模態預測模型，首次執行可能需要下載 FinBERT... "):
                stock_df = get_stock_data(ai_target, data_source=ai_source, years=ai_years)
                if stock_df.empty:
                    raise ValueError("無法取得歷史資料，請確認代碼與資料來源。")
                sentiment_df = get_news_sentiment(ai_target, stock_df.index)
                feature_df = prepare_feature_data(stock_df, sentiment_df)
                (cluster_series, state_series, latest_cluster, latest_state, cluster_summary, _, _) = get_market_state(feature_df)
                feature_df["Market_Cluster"] = cluster_series.reindex(feature_df.index)
                feature_df["Market_State"] = state_series.reindex(feature_df.index)
                X, y, clean_df, feature_scaler, target_scaler = create_tcn_dataset(feature_df, lookback=LOOKBACK_DAYS, horizon=FORECAST_DAYS)
                model, history, validation_data = train_tcn_model(X, y, lookback=LOOKBACK_DAYS, horizon=FORECAST_DAYS)
                metrics = evaluate_tcn(model, validation_data, target_scaler, horizon=FORECAST_DAYS)
                forecast_df = predict_future_ohlc(model, clean_df, feature_scaler, target_scaler, lookback=LOOKBACK_DAYS, horizon=FORECAST_DAYS)
                st.session_state["ai_result"] = {
                    "stock_df": stock_df, "feature_df": feature_df, "clean_df": clean_df,
                    "sentiment_df": sentiment_df, "latest_state": latest_state, "latest_cluster": latest_cluster,
                    "cluster_summary": cluster_summary, "metrics": metrics, "forecast_df": forecast_df,
                    "history": history.history, "source": ai_source_label, "target": ai_target, "years": ai_years
                }
        except Exception as e:
            st.error(f"❌ Version 5 執行失敗：{e}")
            st.info("提示：TCN 建議使用 3 年以上資料；FinBERT 首次執行需要下載模型。")

    if st.session_state.get("ai_result") is not None:
        result = st.session_state["ai_result"]
        stock_df = result["stock_df"]
        forecast_df = result["forecast_df"]
        metrics = result["metrics"]
        latest_state = result["latest_state"]
        latest_cluster = result["latest_cluster"]
        state_icon = {"上升型": "🟢", "下降型": "🔴", "震盪型": "🟡"}.get(latest_state, "⚪")
        last_close = float(stock_df["Close"].iloc[-1])
        first_forecast_close = float(forecast_df["Close"].iloc[0])
        final_forecast_close = float(forecast_df["Close"].iloc[-1])
        month_return = (final_forecast_close / last_close - 1) * 100

        st.success(f"✅ AI 預測完成！標的：**{result['target']}** | 資料來源：{result['source']} | 歷史資料：{result['years']} 年")
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("目前收盤價", f"{last_close:.2f}")
        with r2: st.metric("KMeans 市場狀態", f"{state_icon} {latest_state}")
        with r3: st.metric("首個預測收盤", f"{first_forecast_close:.2f}", f"{(first_forecast_close / last_close - 1) * 100:+.2f}%")
        with r4: st.metric("22 日預測報酬", f"{month_return:+.2f}%")

        with st.container(border=True):
            st.subheader("📊 歷史 K 線 + 未來 22 交易日 AI 預測")
            #
            forecast_days = st.slider("未來預測顯示天數", min_value=1, max_value=FORECAST_DAYS, value=22, key="ai_forecast_days")
            
            # K 線總共固定顯示 44 個交易日
            TOTAL_DISPLAY_DAYS = 44
            
            # 預測幾天，就相應減少歷史天數
            hist_days = TOTAL_DISPLAY_DAYS - forecast_days
            
            historical = stock_df[["Open", "High", "Low", "Close", "Volume"]].tail(hist_days).copy()

            display_forecast = forecast_df.head(forecast_days)
            
            # 合併歷史與預測的所有交易日期
            all_trading_dates = historical.index.union(forecast_df.index)
            # 建立完整日曆日期
            dt_all = pd.date_range(start=all_trading_dates.min(), end=all_trading_dates.max(), freq="D")
            
            # 找出沒有交易的日期
            dt_breaks = list(set(dt_all.strftime("%Y-%m-%d")) - set(all_trading_dates.strftime("%Y-%m-%d")))

            fig_ai = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28], row_titles=["價格", "成交量"])
            fig_ai.add_trace(go.Candlestick(x=historical.index, open=historical["Open"], high=historical["High"], low=historical["Low"], close=historical["Close"], name="歷史 K 線", increasing_line_color="red", increasing_fillcolor="red", decreasing_line_color="green", decreasing_fillcolor="green"), row=1, col=1)
            fig_ai.add_trace(go.Bar(x=historical.index, y=historical["Volume"], name="歷史成交量", marker_color=["red" if c >= o else "green" for c, o in zip(historical["Close"], historical["Open"])], opacity=0.65), row=2, col=1)
            # 以 Scatter + Bar 手動畫出 Version 5 的藍漲 / 橘跌預測 K 線
            for i, (dt, row) in enumerate(display_forecast.iterrows()):
                color = "blue" if row["Close"] >= row["Open"] else "orange"
                fig_ai.add_trace(go.Scatter(x=[dt, dt], y=[row["Low"], row["High"]], mode="lines", line=dict(color=color, width=2), showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig_ai.add_trace(go.Bar(x=[dt], y=[row["Close"] - row["Open"]], base=[min(row["Open"], row["Close"])], width=0.55 * 24 * 60 * 60 * 1000, marker_color=color, name="AI 預測" if i == 0 else None, showlegend=(i == 0), hovertemplate=f"日期: {dt.strftime('%Y-%m-%d')}<br>開盤: {row['Open']:.2f}<br>最高: {row['High']:.2f}<br>最低: {row['Low']:.2f}<br>收盤: {row['Close']:.2f}<br>預測報酬: {row['Return'] * 100:+.2f}%<extra></extra>"), row=1, col=1)
            fig_ai.add_vline(x=display_forecast.index[0], line_dash="dash", line_color="gray", row=1, col=1)
            fig_ai.update_layout(title=f"{result['target']} 歷史 K 線與 AI 預測 | KMeans：{latest_state}", height=680, template="plotly_white", hovermode="x unified", xaxis_rangeslider_visible=False, xaxis=dict(rangebreaks=[dict(values=dt_breaks)]))
            fig_ai.update_yaxes(title_text="價格", row=1, col=1)
            fig_ai.update_yaxes(title_text="成交量", row=2, col=1)
            st.plotly_chart(fig_ai, use_container_width=True)
            st.caption("🔴🟢 歷史 K 線　　🔵 AI 預測上漲　　🟠 AI 預測下跌　　虛線為歷史 / 預測分界")

        with st.container(border=True):
            st.subheader("🔮 未來 22 交易日預測明細")
            output = forecast_df.copy()
            output.insert(0, "日期", output.index.strftime("%Y-%m-%d"))
            output["預測漲跌幅(%)"] = output["Return"] * 100
            output = output.rename(columns={"Open": "預測開盤", "High": "預測最高", "Low": "預測最低", "Close": "預測收盤"})
            output = output[["日期", "預測開盤", "預測最高", "預測最低", "預測收盤", "預測漲跌幅(%)"]]
            st.dataframe(output, use_container_width=True, hide_index=True, column_config={"預測開盤": st.column_config.NumberColumn(format="%.2f"), "預測最高": st.column_config.NumberColumn(format="%.2f"), "預測最低": st.column_config.NumberColumn(format="%.2f"), "預測收盤": st.column_config.NumberColumn(format="%.2f"), "預測漲跌幅(%)": st.column_config.NumberColumn(format="%+.2f %%")})
        
        state_c1, state_c2 = st.columns([1, 1.6])
        # with state_c1:
            # with st.container(border=True):
                # st.subheader("🧭 KMeans 市場型態")
                # st.metric("目前市場型態", f"{state_icon} {latest_state}")
                # st.dataframe( result["cluster_summary"].round(4), use_container_width=True)
        with state_c2:
            with st.container(border=True):
                st.subheader("📰 FinBERT 新聞情緒")
                sentiment = result["sentiment_df"]
                sentiment_score = float(sentiment["sentiment_score"].iloc[-1]) if not sentiment.empty else 0.0
                raw_score = float(sentiment["sentiment_score_raw"].iloc[-1]) if not sentiment.empty else 0.0
                s1, s2, s3 = st.columns(3)
                s1.metric("最新情緒分數", f"{sentiment_score:+.3f}")
                s2.metric("最新原始情緒", f"{raw_score:+.3f}")
                s3.metric("情緒方向", "🟢 正向" if sentiment_score > 0.05 else ("🔴 負向" if sentiment_score < -0.05 else "🟡 中立"))
                sentiment_plot = sentiment.tail(120)
                fig_sent = go.Figure(go.Scatter(x=sentiment_plot.index, y=sentiment_plot["sentiment_score"], mode="lines", name="20日平滑情緒"))
                fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_sent.update_layout(height=300, template="plotly_white", yaxis_title="Sentiment", xaxis_title="日期")
                st.plotly_chart(fig_sent, use_container_width=True)

        with st.container(border=True):
            st.subheader("📏 TCN 驗證集模型評估")
            metric_df = pd.DataFrame([{"預測欄位": n, "MAE": v["MAE"], "RMSE": v["RMSE"]} for n, v in metrics.items()])
            st.dataframe(metric_df, use_container_width=True, hide_index=True, column_config={"MAE": st.column_config.NumberColumn("MAE", format="%.4f"), "RMSE": st.column_config.NumberColumn("RMSE", format="%.4f")})
            history_obj = result.get("history", {})
            if history_obj and "loss" in history_obj:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history_obj["loss"], mode="lines", name="Training Loss"))
                if "val_loss" in history_obj: fig_loss.add_trace(go.Scatter(y=history_obj["val_loss"], mode="lines", name="Validation Loss"))
                fig_loss.update_layout(title="TCN 訓練 / 驗證 Loss", height=320, template="plotly_white", xaxis_title="Epoch", yaxis_title="MSE Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
