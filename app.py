import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta, time as dt_time

# [Added] 自動刷新套件
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# [Added] 引入 PyGithub
try:
    from github import Github, Auth
except ImportError:
    pass 

# ==========================================
# 1. 網頁設定與全域參數
# ==========================================
st.set_page_config(page_title="AI 交易訊號戰情室 (Pro)", layout="wide", initial_sidebar_state="expanded")

# [Fix] CSS 美化：調整按鈕樣式與版面
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important; 
            padding-bottom: 5rem;
            max-width: 98% !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stMetricValue"] {font-size: 24px;}
        /* 讓兩列按鈕更好看 */
        .stButton button {
            width: 100%;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# 2026 年月結算日清單
SETTLEMENT_DATES_2026 = {
    '2026-01-21', '2026-02-18', '2026-03-18', '2026-04-15', '2026-05-20', '2026-06-17',
    '2026-07-15', '2026-08-19', '2026-09-16', '2026-10-21', '2026-11-18', '2026-12-16'
}

HIST_FILE_DAY = 'history_data_day.csv'
HIST_FILE_FULL = 'history_data_full.csv'

# 初始化 Session State (用於記住目前的顯示狀態)
if 'df_view' not in st.session_state: st.session_state.df_view = pd.DataFrame()
if 'entry_idx' not in st.session_state: st.session_state.entry_idx = -1
if 'current_mode' not in st.session_state: st.session_state.current_mode = None # 'day' or 'full'
if 'last_update' not in st.session_state: st.session_state.last_update = None

# ==========================================
# 2. 核心功能: 資料抓取與計算
# ==========================================
class DataEngine:
    def __init__(self):
        self.feature_cols = [
            'Bandwidth', 'MA_Slope', 'Bandwidth_Rate', 'Rel_Volume',
            'K', 'D', 'Position_in_Channel', 'Volatility', 
            'K_Strength', 'Body_Ratio', 'Week', 'Settlement_Day', 'Time_Segment'
        ]
        self.exit_feature_cols = self.feature_cols + ['Floating_PnL', 'Hold_Bars']

    def fetch_realtime_from_anue(self):
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        to_ts = int(datetime.now().timestamp())
        
        # 抓取 1000 筆確保指標運算
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 1000}
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://stock.cnyes.com/market/{symbol}"}
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=8)
            data = res.json().get('data', {})
            if data.get('s') == 'ok' and data.get('t'):
                df = pd.DataFrame({
                    'Time': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
                })
                # 時間校正 (+5分)
                df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
                df['Time'] = df['Time'] + timedelta(minutes=5)
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
                return df
        except Exception as e:
            st.error(f"API Error: {e}")
        return pd.DataFrame()

    def filter_day_session(self, df):
        if df.empty: return df
        df = df.set_index('Time').sort_index()
        return df.between_time(dt_time(8, 50), dt_time(13, 45)).reset_index()

    def calculate_indicators(self, df, mode='day'):
        if df.empty: return df
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']; H = df['High']; L = df['Low']; O = df['Open']; V = df['Volume']
        
        # 指標運算
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        
        # [Fix] 這裡不填 0，保留 NaN 給繪圖用
        df['UB'] = ma20 + 2 * std20
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        df['MA_Slope'] = np.sign(ma20.diff()) # 這裡可以有 NaN
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        df['Rel_Volume'] = V / V.rolling(5).mean()
        
        # KD
        rsv = (C - L.rolling(36).min()) / (H.rolling(36).max() - L.rolling(36).min())
        df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
        
        df['Position_in_Channel'] = (C - df['LB']) / df['Bandwidth']
        df['Volatility'] = (H - L) / C * 100
        df['K_Strength'] = (C - O) / O * 100
        df['Body_Ratio'] = (C - O).abs() / (H - L).replace(0, 1)
        df['Week'] = df['Time'].dt.weekday + 1
        
        if mode == 'full':
            df['Settlement_Day'] = 0
            df['Time_Segment'] = 1
        else:
            df['Settlement_Day'] = df['Time'].apply(lambda t: 1 if (t.weekday() in [2,4] or str(t.date()) in SETTLEMENT_DATES_2026) else 0)
            hm = df['Time'].dt.hour * 100 + df['Time'].dt.minute
            df['Time_Segment'] = np.select([hm <= 930, hm <= 1200], [0, 1], default=2)
        
        # [Critical Fix] 只針對「模型特徵」填補 0，避免模型報錯；但保留 UB/LB/Close 為 NaN 或原值，避免圖表亂掉
        # 使用 bfill 先補前面的 NaN，再用 0 補剩下的
        df[self.feature_cols] = df[self.feature_cols].fillna(method='bfill').fillna(0)
        
        return df

# ==========================================
# 3. 策略引擎
# ==========================================
class StrategyEngine:
    def __init__(self, models, params, df):
        self.models = models
        self.params = params
        self.df = df
        self.processor = DataEngine()

    def find_entry_info(self, entry_time_obj):
        if entry_time_obj is None: return -1, 0.0
        time_str = entry_time_obj.strftime("%H:%M")
        matches = self.df[self.df['Time'].astype(str).str.contains(time_str, na=False)]
        if not matches.empty:
            return matches.index[-1], matches.iloc[-1]['Close']
        return -1, 0.0

    def run_analysis(self, user_pos_type, entry_time_obj):
        if self.df.empty: return pd.DataFrame(), {}
        
        history_records = []
        X_all = self.df[self.processor.feature_cols]
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        user_entry_idx, user_cost = self.find_entry_info(entry_time_obj) if u_pos != "Empty" else (-1, 0.0)
        
        s_pos, s_price, s_idx = 0, 0.0, 0
        
        for i in range(len(self.df)):
            curr_row = self.df.iloc[i]
            curr_feats = X_all.iloc[[i]]
            
            try:
                p_long = self.models['Long_Entry_Model'].predict_proba(curr_feats)[0][1]
                p_short = self.models['Short_Entry_Model'].predict_proba(curr_feats)[0][1]
            except: p_long, p_short = 0.0, 0.0
            
            trend = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            s_action, s_detail = "⚪ 觀望", trend
            
            # --- 策略訊號 ---
            if s_pos == 0:
                if p_long > self.params['entry'] and p_long > p_short:
                    s_pos, s_price, s_idx, s_action, s_detail = 1, curr_row['Close'], i, "🔴 買進", f"多 {p_long:.0%} {trend}"
                elif p_short > self.params['entry'] and p_short > p_long:
                    s_pos, s_price, s_idx, s_action, s_detail = -1, curr_row['Close'], i, "🟢 放空", f"空 {p_short:.0%} {trend}"
            elif s_pos == 1:
                pnl = curr_row['Close'] - s_price
                if pnl <= -self.params['stop']:
                    s_pos, s_action, s_detail = 0, "💥 停損", f"損 {pnl:.0f}"
                else:
                    curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=i-s_idx)
                    ep = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                    s_action, s_detail = ("❌ 多出", f"帳{pnl:.0f}(出:{ep:.0%})") if ep > self.params['exit'] else ("⏳ 續抱", f"帳{pnl:.0f}(續:{1-ep:.0%})")
                    if ep > self.params['exit']: s_pos = 0
            elif s_pos == -1:
                pnl = s_price - curr_row['Close']
                if pnl <= -self.params['stop']:
                    s_pos, s_action, s_detail = 0, "💥 停損", f"損 {pnl:.0f}"
                else:
                    curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=i-s_idx)
                    ep = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                    s_action, s_detail = ("❎ 空出", f"帳{pnl:.0f}(出:{ep:.0%})") if ep > self.params['exit'] else ("⏳ 續抱", f"帳{pnl:.0f}(續:{1-ep:.0%})")
                    if ep > self.params['exit']: s_pos = 0

            # --- 真實部位建議 ---
            u_action, u_note = "-", "-"
            if u_pos != "Empty" and i >= user_entry_idx:
                hold_bars = i - user_entry_idx
                if u_pos == "Long":
                    pnl = curr_row['Close'] - user_cost
                    if i == user_entry_idx: u_action, u_note = "🔴 多單進場", f"本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        u_action = "❌ 出場" if ep > self.params['exit'] else ("🔥 加碼" if p_long > self.params['entry'] else "⏳ 續抱")
                        u_note = f"帳{pnl:.0f}(出:{ep:.0%})"
                elif u_pos == "Short":
                    pnl = user_cost - curr_row['Close']
                    if i == user_entry_idx: u_action, u_note = "🟢 空單進場", f"本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        u_action = "❎ 出場" if ep > self.params['exit'] else ("🔥 加碼" if p_short > self.params['entry'] else "⏳ 續抱")
                        u_note = f"帳{pnl:.0f}(出:{ep:.0%})"

            history_records.append({
                'Time': curr_row['Time'], 'Close': curr_row['Close'],
                'UB': curr_row.get('UB', np.nan), 'LB': curr_row.get('LB', np.nan),
                'Strategy_Action': s_action, 'Strategy_Detail': s_detail,
                'User_Advice': u_action, 'User_Note': u_note
            })
            
        return pd.DataFrame(history_records), user_entry_idx

# ==========================================
# 4. GitHub 存檔功能
# ==========================================
def push_to_github(file_path, df_to_save):
    token = st.secrets.get("GITHUB_TOKEN")
    repo_name = st.secrets.get("GITHUB_REPO")
    if not token or not repo_name: return "❌ 請設定 Secrets"
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        csv_content = df_to_save.to_csv(index=False)
        try:
            repo.update_file(repo.get_contents(file_path).path, f"Update {file_path}", csv_content, repo.get_contents(file_path).sha)
        except:
            repo.create_file(file_path, f"Create {file_path}", csv_content)
        return "✅ 雲端存檔成功！"
    except Exception as e: return f"❌ 失敗: {e}"

# ==========================================
# 5. UI 主程式
# ==========================================
@st.cache_resource
def load_models():
    try:
        models = {}
        for name in ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']:
            if os.path.exists(f"models/{name}.pkl"): models[name] = joblib.load(f"models/{name}.pkl")
            elif os.path.exists(f"{name}.pkl"): models[name] = joblib.load(f"{name}.pkl")
        return models if len(models)==4 else None
    except: return None

# 自動刷新
if st_autorefresh: st_autorefresh(interval=300000, key="auto_refresh")

engine = DataEngine()
models = load_models()

# --- 側邊欄 ---
with st.sidebar:
    st.header("🎮 控制台")
    
    # [Fix] 按鈕分兩列，整潔排列
    col_day, col_full = st.columns(2)
    trigger_day = col_day.button("🌞 更新日盤", type="primary", use_container_width=True)
    trigger_full = col_full.button("🌙 更新全盤", use_container_width=True)
    
    # 策略參數
    with st.expander("⚙️ 參數與部位", expanded=True):
        p_entry = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        p_exit = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        p_stop = st.number_input("硬停損", 100, step=10)
        st.markdown("---")
        u_pos = st.radio("真實持倉", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"])
        u_time = st.time_input("進場時間", value=dt_time(9,0), step=300) if u_pos != "空手 (Empty)" else None

    # 歷史資料管理 (收納起來，比較整潔)
    with st.expander("💾 歷史資料庫管理", expanded=False):
        tab_h_day, tab_h_full = st.tabs(["日盤歷史", "全盤歷史"])
        
        def handle_history(file_path, key_suffix):
            up = st.file_uploader(f"上傳覆蓋 ({key_suffix})", type=['csv'], key=f"up_{key_suffix}")
            if up:
                pd.read_csv(up).to_csv(file_path, index=False)
                st.success("已更新本地檔")
            if st.button(f"☁️ 寫入 GitHub ({key_suffix})", key=f"save_{key_suffix}"):
                if os.path.exists(file_path):
                    st.write(push_to_github(file_path, pd.read_csv(file_path)))
                else: st.error("無本地檔")

        with tab_h_day: handle_history(HIST_FILE_DAY, "day")
        with tab_h_full: handle_history(HIST_FILE_FULL, "full")

# --- 資料處理邏輯 ---
def process_data(mode):
    hist_file = HIST_FILE_DAY if mode == 'day' else HIST_FILE_FULL
    df_hist = pd.read_csv(hist_file) if os.path.exists(hist_file) else pd.DataFrame()
    if not df_hist.empty: df_hist['Time'] = pd.to_datetime(df_hist['Time'])
    
    df_real = engine.fetch_realtime_from_anue()
    
    if not df_real.empty:
        df_total = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time', keep='last').sort_values('Time')
    else:
        df_total = df_hist

    if df_total.empty: return pd.DataFrame(), "無資料"

    if mode == 'day':
        df_calc = engine.filter_day_session(df_total)
    else:
        df_calc = df_total
        
    # 計算指標
    df_calc = engine.calculate_indicators(df_calc, mode=mode)
    
    # 存檔邏輯: 只存原始 OHLCV 回歷史檔 (不存指標)
    if not df_total.empty:
        save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        if mode == 'day':
            # 日盤歷史檔只存日盤資料
            engine.filter_day_session(df_total)[save_cols].to_csv(hist_file, index=False)
        else:
            df_total[save_cols].to_csv(hist_file, index=False)
            
    return df_calc, "OK"

# --- 觸發更新邏輯 ---
if trigger_day:
    with st.spinner("更新日盤中..."):
        df_res, status = process_data('day')
        if status == "OK":
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'day'
            st.session_state.last_update = datetime.now()
        else: st.error(status)

if trigger_full:
    with st.spinner("更新全盤中..."):
        df_res, status = process_data('full')
        if status == "OK":
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'full'
            st.session_state.last_update = datetime.now()
        else: st.error(status)

# --- 主畫面顯示 ---
if not st.session_state.df_view.empty and models:
    mode_name = "🌞 日盤" if st.session_state.current_mode == 'day' else "🌙 全盤"
    st.title(f"{mode_name}戰情室")
    st.caption(f"最後更新: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else '-'}")
    
    # 執行策略
    strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, st.session_state.df_view)
    df_display, entry_idx = strat.run_analysis(u_pos, u_time)
    
    # [Fix] 顯示範圍設定 (避免圖表擠成一團)
    # 取最後 150 根 K 棒來畫圖，但保留完整 DataFrame 供縮放
    df_chart = df_display.copy()
    df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
    
    total_len = len(df_chart)
    default_range_start = max(0, total_len - 150)
    
    # 繪圖
    fig = go.Figure()
    
    # 布林通道 (處理 NaN 不顯示的問題)
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['UB'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['LB'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', name='BB'))
    
    # K線/收盤價
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
    
    # 訊號
    for action, symbol, color, name in [('買進', 'triangle-up', 'red', 'Buy'), ('放空', 'triangle-down', 'green', 'Sell'), ('出', 'x', 'gray', 'Exit')]:
        mask = df_chart['Strategy_Action'].str.contains(action)
        if mask.any():
            subset = df_chart[mask]
            fig.add_trace(go.Scatter(x=subset['Time'].dt.strftime('%H:%M'), y=subset['Close'], mode='markers', marker=dict(symbol=symbol, size=12, color=color), name=name))

    # 使用者進場點
    if entry_idx != -1 and entry_idx in df_chart.index:
        row = df_chart.loc[entry_idx]
        fig.add_trace(go.Scatter(x=[row['Time_Str']], y=[row['Close']], mode='markers', marker=dict(symbol='star', size=18, color='gold', line=dict(width=1, color='black')), name='My Entry'))

    fig.update_layout(
        height=500, margin=dict(t=30, l=0, r=0, b=0),
        xaxis=dict(type='category', rangeslider=dict(visible=True), range=[default_range_start, total_len-1]),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # [Fix] 訊號履歷 (一致性優化)
    st.subheader("📜 訊號履歷")
    st.dataframe(
        df_display.iloc[::-1],
        height=400,
        column_config={
            "Time": st.column_config.DatetimeColumn("時間", format="MM-dd HH:mm", width="small"),
            "Close": st.column_config.NumberColumn("價位", format="%d", width="small"),
            "Strategy_Action": st.column_config.TextColumn("策略", width="small"),
            "Strategy_Detail": st.column_config.TextColumn("多空機率", width="medium"),
            "User_Advice": st.column_config.TextColumn("建議", width="small"),
            "User_Note": st.column_config.TextColumn("持倉損益", width="medium"),
            "UB": None, "LB": None # 隱藏欄位
        },
        use_container_width=True,
        hide_index=True
    )
    
elif models is None:
    st.warning("⚠️ 請確認 models/ 資料夾內是否有 4 個 .pkl 模型檔")
else:
    st.info("👈 請點擊左側「🌞 更新日盤」或「🌙 更新全盤」開始分析")
