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

# [Added] 引入 yfinance 作為備援資料源
try:
    import yfinance as yf
except ImportError:
    yf = None

# ==========================================
# 1. 網頁設定與全域參數
# ==========================================
st.set_page_config(page_title="AI 交易訊號戰情室 (Pro)", layout="wide", initial_sidebar_state="expanded")

# CSS 美化
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

MASTER_HIST_FILE = 'history_data_full.csv' 

if 'df_view' not in st.session_state: st.session_state.df_view = pd.DataFrame()
if 'entry_idx' not in st.session_state: st.session_state.entry_idx = -1
if 'current_mode' not in st.session_state: st.session_state.current_mode = None 
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

    def _parse_api_response(self, data):
        if not data.get('t'): return pd.DataFrame()
        df = pd.DataFrame({
            'Time': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
        })
        # UTC -> Taiwan -> +5min
        df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
        df['Time'] = df['Time'] + timedelta(minutes=5)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
        return df

    def fetch_yahoo_fallback(self):
        """[Backup] 使用 Yahoo Finance 抓取日盤資料"""
        if yf is None: return pd.DataFrame()
        try:
            # TXF=F 是台指期代碼
            ticker = yf.Ticker("TXF=F")
            # 抓取最近 2 天，5分K
            df = ticker.history(period="2d", interval="5m")
            if df.empty: return pd.DataFrame()
            
            df = df.reset_index()
            # 處理時區
            if df['Datetime'].dt.tz is None:
                df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
            else:
                df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Taipei')
            
            # 格式統一
            df = df.rename(columns={'Datetime': 'Time'})
            df['Time'] = df['Time'].dt.tz_localize(None) # 移除時區資訊以便合併
            # Yahoo 的時間是 K 棒開始時間，我們習慣用結束時間 (+5min)
            df['Time'] = df['Time'] + timedelta(minutes=5)
            
            df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except:
            return pd.DataFrame()

    def fetch_realtime_from_anue(self):
        """雙引擎抓取：Anue 優先，若缺日盤則呼叫 Yahoo 救援"""
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://stock.cnyes.com/market/{symbol}"}
        
        # 1. 嘗試 Anue
        to_ts = int(datetime.now().timestamp())
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 1000}
        
        df_final = pd.DataFrame()
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=5)
            data = res.json().get('data', {})
            if data.get('t'):
                df_final = self._parse_api_response(data)
                
                # 2. [斷層偵測]
                # 如果抓回來的資料都是 15:00 以後的 (夜盤)，但現在時間超過晚上，代表早上的日盤丟了
                if not df_final.empty:
                    min_hour = df_final['Time'].dt.hour.min()
                    # 如果最早的一筆資料已經是下午 3 點以後
                    if min_hour >= 15:
                        # 3. [啟動 Yahoo 救援]
                        st.toast("偵測到 Anue 日盤資料缺失，啟動 Yahoo Finance 救援...", icon="🚑")
                        df_yahoo = self.fetch_yahoo_fallback()
                        
                        if not df_yahoo.empty:
                            # 過濾出 Yahoo 的日盤資料 (08:45 ~ 13:45)
                            mask_day = (df_yahoo['Time'].dt.hour >= 8) & (df_yahoo['Time'].dt.hour < 14)
                            df_yahoo_day = df_yahoo[mask_day]
                            
                            # 合併：Yahoo日盤 + Anue夜盤
                            df_final = pd.concat([df_yahoo_day, df_final]).drop_duplicates(subset='Time', keep='last').sort_values('Time')
                            st.toast(f"成功補回 {len(df_yahoo_day)} 筆日盤資料！", icon="✅")
            
            return df_final

        except Exception as e:
            st.error(f"API Error: {e}")
            return df_final

    def filter_day_session(self, df):
        if df.empty: return df
        df = df.set_index('Time').sort_index()
        return df.between_time(dt_time(8, 45), dt_time(13, 50)).reset_index()

    def calculate_indicators(self, df, mode='day'):
        if df.empty: return df
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']; H = df['High']; L = df['Low']; O = df['Open']; V = df['Volume']
        
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        
        df['UB'] = ma20 + 2 * std20
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        df['MA_Slope'] = np.sign(ma20.diff()) 
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        df['Rel_Volume'] = V / V.rolling(5).mean()
        
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

with st.sidebar:
    st.header("🎮 控制台")
    col_day, col_full = st.columns(2)
    trigger_day = col_day.button("🌞 更新日盤", type="primary", use_container_width=True)
    trigger_full = col_full.button("🌙 更新全盤", use_container_width=True)
    
    with st.expander("⚙️ 參數與部位", expanded=True):
        p_entry = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        p_exit = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        p_stop = st.number_input("硬停損", 100, step=10)
        st.markdown("---")
        u_pos = st.radio("真實持倉", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"])
        u_time = st.time_input("進場時間", value=dt_time(9,0), step=300) if u_pos != "空手 (Empty)" else None

    with st.expander("💾 資料庫管理 (Master)", expanded=False):
        up = st.file_uploader("上傳歷史檔 (覆蓋)", type=['csv'])
        if up:
            pd.read_csv(up).to_csv(MASTER_HIST_FILE, index=False)
            st.success("已更新主資料庫")
        if st.button("☁️ 寫入 GitHub", key="save_master"):
            if os.path.exists(MASTER_HIST_FILE):
                st.write(push_to_github(MASTER_HIST_FILE, pd.read_csv(MASTER_HIST_FILE)))
            else: st.error("無本地檔")

def process_data(mode):
    df_hist = pd.read_csv(MASTER_HIST_FILE) if os.path.exists(MASTER_HIST_FILE) else pd.DataFrame()
    if not df_hist.empty: df_hist['Time'] = pd.to_datetime(df_hist['Time'])
    
    # [Smart Fetch] 自動抓取 + 自動回補 (Anue + Yahoo)
    df_real = engine.fetch_realtime_from_anue()
    
    if not df_real.empty:
        df_total = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time', keep='last').sort_values('Time')
        # 存回主檔
        save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_total[save_cols].to_csv(MASTER_HIST_FILE, index=False)
    else:
        df_total = df_hist

    if df_total.empty: return pd.DataFrame(), "無資料"

    if mode == 'day':
        df_calc = engine.filter_day_session(df_total)
    else:
        df_calc = df_total
        
    if df_calc.empty: return pd.DataFrame(), "該時段無資料 (建議先更新全盤以累積歷史)"

    df_calc = engine.calculate_indicators(df_calc, mode=mode)
    return df_calc, "OK"

if trigger_day:
    with st.spinner("整合日盤數據中..."):
        df_res, status = process_data('day')
        if status == "OK":
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'day'
            st.session_state.last_update = datetime.now()
        else: st.error(status)

if trigger_full:
    with st.spinner("整合全盤數據中..."):
        df_res, status = process_data('full')
        if status == "OK":
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'full'
            st.session_state.last_update = datetime.now()
        else: st.error(status)

if not st.session_state.df_view.empty and models:
    mode_name = "🌞 日盤" if st.session_state.current_mode == 'day' else "🌙 全盤"
    st.title(f"{mode_name}戰情室")
    st.caption(f"最後更新: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else '-'}")
    
    if len(st.session_state.df_view) < 50:
        st.warning(f"⚠️ 資料筆數 ({len(st.session_state.df_view)}) 不足，技術指標可能偏差。")

    strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, st.session_state.df_view)
    df_display, entry_idx = strat.run_analysis(u_pos, u_time)
    
    df_chart = df_display.copy()
    df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
    total_len = len(df_chart)
    default_range_start = max(0, total_len - 150)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['UB'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['LB'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', name='BB'))
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
    
    for action, symbol, color, name in [('買進', 'triangle-up', 'red', 'Buy'), ('放空', 'triangle-down', 'green', 'Sell'), ('出', 'x', 'gray', 'Exit')]:
        mask = df_chart['Strategy_Action'].str.contains(action)
        if mask.any():
            subset = df_chart[mask]
            fig.add_trace(go.Scatter(x=subset['Time'].dt.strftime('%H:%M'), y=subset['Close'], mode='markers', marker=dict(symbol=symbol, size=12, color=color), name=name))

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
            "UB": None, "LB": None
        },
        use_container_width=True,
        hide_index=True
    )
    
elif models is None:
    st.warning("⚠️ 請確認 models/ 資料夾內是否有 4 個 .pkl 模型檔")
else:
    st.info("👈 請點擊左側「🌞 更新日盤」或「🌙 更新全盤」")
