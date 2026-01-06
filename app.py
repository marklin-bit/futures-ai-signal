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
    from github import Github, Auth, GithubException
except ImportError:
    pass 

# ==========================================
# 1. 網頁設定與全域參數
# ==========================================
st.set_page_config(
    page_title="AI 交易訊號戰情室 (Pro)", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# CSS 美化
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem !important; 
            padding-bottom: 3rem;
            max-width: 98% !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px;
            font-weight: bold;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
        div[data-testid="stDataFrame"] {
            font-family: 'Consolas', 'Monaco', monospace;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# [重要] 結算日設定
# ------------------------------------------------------------------
SETTLEMENT_DATES = {
    # 2025
    '2025-01-15', '2025-02-19', '2025-03-19', '2025-04-16', '2025-05-21', '2025-06-18',
    '2025-07-16', '2025-08-20', '2025-09-17', '2025-10-15', '2025-11-19', '2025-12-17',
    # 2026
    '2026-01-21', '2026-02-18', '2026-03-18', '2026-04-15', '2026-05-20', '2026-06-17',
    '2026-07-15', '2026-08-19', '2026-09-16', '2026-10-21', '2026-11-18', '2026-12-16'
}

# 資料庫路徑
HIST_FILE_DAY = 'history_data_day.csv'
HIST_FILE_FULL = 'history_data_full.csv'

# Session State
if 'df_view' not in st.session_state: st.session_state.df_view = pd.DataFrame()
if 'entry_idx' not in st.session_state: st.session_state.entry_idx = -1
if 'current_mode' not in st.session_state: st.session_state.current_mode = None 
if 'last_update' not in st.session_state: st.session_state.last_update = None
if 'data_range_info' not in st.session_state: st.session_state.data_range_info = ""

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

    def _parse_anue_response(self, data):
        if not data.get('t'): return pd.DataFrame()
        try:
            df = pd.DataFrame({
                'Time': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
            })
            df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
            df['Time'] = df['Time'] + timedelta(minutes=5)
            
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=cols)
            return df
        except Exception as e:
            st.warning(f"資料解析異常: {e}")
            return pd.DataFrame()

    def fetch_anue_raw(self):
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://stock.cnyes.com/market/{symbol}"}
        
        to_ts = int(datetime.now().timestamp())
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 1000}
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json().get('data', {})
                return self._parse_anue_response(data)
        except Exception as e:
            st.error(f"鉅亨網連線錯誤: {e}")
        return pd.DataFrame()

    def merge_and_save(self, api_df, hist_file, is_day_mode=False):
        # 讀取歷史資料
        hist_df = pd.DataFrame()
        if os.path.exists(hist_file):
            try:
                hist_df = pd.read_csv(hist_file)
                hist_df['Time'] = pd.to_datetime(hist_df['Time'])
            except: pass

        # 過濾新 API 資料
        new_df = api_df.copy()
        if not new_df.empty and is_day_mode:
            new_df = new_df.set_index('Time').sort_index()
            new_df = new_df.between_time(dt_time(8, 45), dt_time(13, 45)).reset_index()

        # 合併邏輯 (確保歷史連續性)
        if not new_df.empty:
            if not hist_df.empty:
                full_df = pd.concat([hist_df, new_df])
            else:
                full_df = new_df
            
            # 重要：合併後先排序，確保時間序列正確，這對後續計算至關重要
            full_df = full_df.sort_values('Time')
            full_df = full_df.drop_duplicates(subset='Time', keep='last').reset_index(drop=True)
        else:
            full_df = hist_df

        # 日盤再次過濾雜訊
        if is_day_mode and not full_df.empty:
             full_df = full_df.set_index('Time').sort_index()
             full_df = full_df.between_time(dt_time(8, 45), dt_time(13, 45)).reset_index()

        # 自動清理：只保留最近 5 個交易日
        if not full_df.empty:
            full_df['Date'] = full_df['Time'].dt.date
            unique_dates = sorted(full_df['Date'].unique())
            if len(unique_dates) > 5:
                cutoff_date = unique_dates[-5]
                full_df = full_df[full_df['Date'] >= cutoff_date]
            full_df = full_df.drop(columns=['Date'])

        # 存檔
        if not full_df.empty:
            full_df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(hist_file, index=False)
            
            start = full_df['Time'].iloc[0].strftime('%m/%d %H:%M')
            end = full_df['Time'].iloc[-1].strftime('%m/%d %H:%M')
            days = len(unique_dates) if 'unique_dates' in locals() else '?'
            st.session_state.data_range_info = f"{start} ~ {end} (共 {len(full_df)} K / {days} 天)"
        else:
            st.session_state.data_range_info = "尚無資料"

        return full_df

    def calculate_indicators(self, df, mode='day'):
        """計算技術指標 (確保無未來數據汙染)"""
        if df.empty: return df
        df = df.copy()
        
        # 確保時間序列由舊到新排列，這對 Rolling 計算是必須的
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']; H = df['High']; L = df['Low']; O = df['Open']; V = df['Volume']
        
        # 指標計算
        # 這裡會用到前面的歷史資料。只要 df 裡包含昨天的資料，今天的 MA 就不會是 NaN
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        df['UB'] = ma20 + 2 * std20
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        df['MA_Slope'] = np.sign(ma20.diff()) 
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        
        vol_ma = V.rolling(5).mean().replace(0, 1)
        df['Rel_Volume'] = V / vol_ma
        
        lowest_l = L.rolling(36).min()
        highest_h = H.rolling(36).max()
        denom = (highest_h - lowest_l).replace(0, 1)
        rsv = (C - lowest_l) / denom
        
        df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
        
        bw_safe = df['Bandwidth'].replace(0, 0.0001)
        df['Position_in_Channel'] = (C - df['LB']) / bw_safe
        
        df['Volatility'] = (H - L) / C * 100
        df['K_Strength'] = (C - O) / O * 100
        df['Body_Ratio'] = (C - O).abs() / (H - L).replace(0, 1)
        df['Week'] = df['Time'].dt.weekday + 1
        
        if mode == 'full':
            df['Settlement_Day'] = 0
            df['Time_Segment'] = 1
        else:
            df['Settlement_Day'] = df['Time'].apply(
                lambda t: 1 if (t.weekday() in [2,4] or str(t.date()) in SETTLEMENT_DATES) else 0
            )
            hm = df['Time'].dt.hour * 100 + df['Time'].dt.minute
            df['Time_Segment'] = np.select([hm <= 930, hm <= 1200], [0, 1], default=2)
        
        # [邏輯確認]
        # 使用 fillna(0) 是為了處理「資料集最開頭」的 NaN (5天前的資料)。
        # 因為我們在計算前已經載入了完整的歷史資料 (df 包含 5 天)，
        # 所以「今天 08:45」的資料前面已經有「昨天」的資料做支撐，
        # 計算出來的 MA, KD 等指標會是有效值，不會變成 0。
        # 這樣既避免了「偷看未來 (bfill)」，也確保了「今日計算連續性」。
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
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
        """
        執行策略分析
        邏輯說明：
        1. Batch Prediction (批次預測) 是為了加速。
        2. 因為 Tree Model 是 Stateless (無狀態) 的，Model.predict(Row_N) 的結果
           只取決於 Row_N 的特徵。
        3. Row_N 的特徵 (如 MA20) 在 DataEngine 階段已經計算完成，
           其數值僅包含 Row_0 到 Row_N 的歷史資訊。
        4. 因此，一次算完所有機率，與迴圈中逐筆計算，數學結果完全相同，且無未來視問題。
        """
        if self.df.empty: return pd.DataFrame(), -1
        
        X_all = self.df[self.processor.feature_cols]
        
        # --- Step 1: 批次預測 (快速算出每根 K 棒的原始訊號) ---
        try:
            # 這裡計算出來的 probs_long[i] 代表：
            # 在第 i 根 K 棒結束當下 (包含了 0~i 的歷史特徵)，AI 對未來的判斷
            probs_long = self.models['Long_Entry_Model'].predict_proba(X_all)[:, 1]
            probs_short = self.models['Short_Entry_Model'].predict_proba(X_all)[:, 1]
        except:
            probs_long = np.zeros(len(self.df))
            probs_short = np.zeros(len(self.df))

        # --- Step 2: 準備使用者持倉資訊 ---
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        user_entry_idx, user_cost = self.find_entry_info(entry_time_obj) if u_pos != "Empty" else (-1, 0.0)
        
        # --- Step 3: 策略狀態機迴圈 (模擬時間推演) ---
        history_records = []
        
        s_pos = 0     # 策略持倉
        s_price = 0.0 # 進場價
        s_idx = 0     # 進場Index
        
        # 逐筆模擬，確保每一筆交易決策都只基於當下或過去的狀態
        for i in range(len(self.df)):
            curr_row = self.df.iloc[i]
            
            # 取出當下時間點 AI 的判斷 (這個機率值只包含 <= i 時間點的資訊)
            p_long = probs_long[i]
            p_short = probs_short[i]
            
            trend_str = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            s_action, s_detail = "⚪ 觀望", trend_str
            
            # 策略進出場邏輯 (狀態機)
            if s_pos == 0:
                if p_long > self.params['entry'] and p_long > p_short:
                    s_pos = 1; s_price = curr_row['Close']; s_idx = i
                    s_action = "🔴 買進"; s_detail = f"多 {p_long:.0%} {trend_str}"
                elif p_short > self.params['entry'] and p_short > p_long:
                    s_pos = -1; s_price = curr_row['Close']; s_idx = i
                    s_action = "🟢 放空"; s_detail = f"空 {p_short:.0%} {trend_str}"
            
            elif s_pos == 1: # 持有多單
                pnl = curr_row['Close'] - s_price
                if pnl <= -self.params['stop']:
                    s_pos = 0; s_action = "💥 停損"; s_detail = f"損 {pnl:.0f}"
                else:
                    # 出場特徵需要包含當下的 PnL，所以這裡需即時組建特徵
                    # 這邊只針對單筆資料預測，也不會偷看未來
                    row_feats = X_all.iloc[[i]].copy()
                    row_feats['Floating_PnL'] = pnl
                    row_feats['Hold_Bars'] = i - s_idx
                    ep = self.models['Long_Exit_Model'].predict_proba(row_feats[self.processor.exit_feature_cols])[0][1]
                    
                    if ep > self.params['exit']:
                        s_pos = 0; s_action = "❌ 多出"; s_detail = f"帳{pnl:.0f}(出:{ep:.0%})"
                    else:
                        s_action = "⏳ 續抱"; s_detail = f"帳{pnl:.0f}(續:{1-ep:.0%})"

            elif s_pos == -1: # 持有空單
                pnl = s_price - curr_row['Close']
                if pnl <= -self.params['stop']:
                    s_pos = 0; s_action = "💥 停損"; s_detail = f"損 {pnl:.0f}"
                else:
                    row_feats = X_all.iloc[[i]].copy()
                    row_feats['Floating_PnL'] = pnl
                    row_feats['Hold_Bars'] = i - s_idx
                    ep = self.models['Short_Exit_Model'].predict_proba(row_feats[self.processor.exit_feature_cols])[0][1]
                    
                    if ep > self.params['exit']:
                        s_pos = 0; s_action = "❎ 空出"; s_detail = f"帳{pnl:.0f}(出:{ep:.0%})"
                    else:
                        s_action = "⏳ 續抱"; s_detail = f"帳{pnl:.0f}(續:{1-ep:.0%})"

            # 真實持倉建議 (User Advice) - 邏輯同上，略為精簡
            u_action, u_note = "-", "-"
            if u_pos != "Empty" and i >= user_entry_idx:
                hold_bars = i - user_entry_idx
                if u_pos == "Long":
                    pnl = curr_row['Close'] - user_cost
                    if i == user_entry_idx: u_action, u_note = "🔴 多單進場", f"本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        row_feats = X_all.iloc[[i]].copy()
                        row_feats['Floating_PnL'] = pnl; row_feats['Hold_Bars'] = hold_bars
                        ep = self.models['Long_Exit_Model'].predict_proba(row_feats[self.processor.exit_feature_cols])[0][1]
                        u_action = "❌ 出場" if ep > self.params['exit'] else ("🔥 加碼" if p_long > self.params['entry'] else "⏳ 續抱")
                        u_note = f"帳{pnl:.0f}(出:{ep:.0%})"
                elif u_pos == "Short":
                    pnl = user_cost - curr_row['Close']
                    if i == user_entry_idx: u_action, u_note = "🟢 空單進場", f"本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        row_feats = X_all.iloc[[i]].copy()
                        row_feats['Floating_PnL'] = pnl; row_feats['Hold_Bars'] = hold_bars
                        ep = self.models['Short_Exit_Model'].predict_proba(row_feats[self.processor.exit_feature_cols])[0][1]
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
    if not token or not repo_name: return "❌ 缺少 GitHub 設定"
    if "/" not in repo_name: return f"❌ Repo 名稱錯誤: '{repo_name}'"

    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        csv_content = df_to_save.to_csv(index=False)
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, f"Update {file_path}", csv_content, contents.sha)
            return "✅ 雲端更新成功！"
        except:
            repo.create_file(file_path, f"Create {file_path}", csv_content)
            return "✅ 雲端建立成功！"
    except GithubException as e:
        if e.status == 404: return "❌ 404 錯誤 (Repo 不存在或無權限)"
        return f"❌ GitHub 錯誤: {e.data.get('message', str(e))}"
    except Exception as e: return f"❌ 未知錯誤: {str(e)}"

# ==========================================
# 5. UI 主程式
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    req = ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']
    missing = []
    for name in req:
        p1, p2 = f"models/{name}.pkl", f"{name}.pkl"
        if os.path.exists(p1): models[name] = joblib.load(p1)
        elif os.path.exists(p2): models[name] = joblib.load(p2)
        else: missing.append(name)
    if missing: st.error(f"❌ 缺少模型: {missing}"); return None
    return models

if st_autorefresh: st_autorefresh(interval=300000, key="auto_refresh")

engine = DataEngine()
models = load_models()

with st.sidebar:
    st.header("🎮 控制台")
    col_day, col_full = st.columns(2)
    trigger_day = col_day.button("🌞 更新日盤", type="primary", use_container_width=True)
    trigger_full = col_full.button("🌙 更新全盤", use_container_width=True)
    
    if st.button("🧹 重置資料庫"):
        if os.path.exists(HIST_FILE_DAY): os.remove(HIST_FILE_DAY)
        if os.path.exists(HIST_FILE_FULL): os.remove(HIST_FILE_FULL)
        st.cache_data.clear()
        st.session_state.df_view = pd.DataFrame()
        st.rerun()

    with st.expander("⚙️ 參數與部位", expanded=True):
        p_entry = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        p_exit = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        p_stop = st.number_input("硬停損", 50, 500, 100, step=10)
        st.markdown("---")
        u_pos = st.radio("真實持倉", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"])
        u_time = st.time_input("進場時間", value=dt_time(9,0), step=300) if u_pos != "空手 (Empty)" else None

    with st.expander("💾 資料庫管理", expanded=False):
        st.caption("手動維護與雲端同步")
        tab_d, tab_f = st.tabs(["日盤", "全盤"])
        with tab_d:
            up_day = st.file_uploader("上傳日盤", type=['csv'], key="up_day")
            if up_day: pd.read_csv(up_day).to_csv(HIST_FILE_DAY, index=False); st.success("更新日盤")
            if st.button("上傳 GitHub (日)", key="gd"):
                if os.path.exists(HIST_FILE_DAY): 
                    with st.spinner("上傳中..."): st.write(push_to_github(HIST_FILE_DAY, pd.read_csv(HIST_FILE_DAY)))
        with tab_f:
            up_full = st.file_uploader("上傳全盤", type=['csv'], key="up_full")
            if up_full: pd.read_csv(up_full).to_csv(HIST_FILE_FULL, index=False); st.success("更新全盤")
            if st.button("上傳 GitHub (全)", key="gf"):
                if os.path.exists(HIST_FILE_FULL): 
                    with st.spinner("上傳中..."): st.write(push_to_github(HIST_FILE_FULL, pd.read_csv(HIST_FILE_FULL)))

def process_data(mode):
    hist_file = HIST_FILE_DAY if mode == 'day' else HIST_FILE_FULL
    api_df = engine.fetch_anue_raw()
    final_df = engine.merge_and_save(api_df, hist_file, is_day_mode=(mode=='day'))
    if final_df.empty: return pd.DataFrame(), "❌ 無資料"
    status = "OK" if not api_df.empty else "⚠️ API 無新資料"
    return engine.calculate_indicators(final_df, mode=mode), status

if trigger_day:
    with st.spinner("整合日盤..."):
        df_res, status = process_data('day')
        st.session_state.df_view = df_res; st.session_state.current_mode = 'day'
        st.session_state.last_update = datetime.now()
        if status != "OK": st.toast(status, icon="⚠️")

if trigger_full:
    with st.spinner("整合全盤..."):
        df_res, status = process_data('full')
        st.session_state.df_view = df_res; st.session_state.current_mode = 'full'
        st.session_state.last_update = datetime.now()
        if status != "OK": st.toast(status, icon="⚠️")

if not st.session_state.df_view.empty and models:
    icon = "🌞" if st.session_state.current_mode == 'day' else "🌙"
    st.title(f"{icon} 戰情室")
    c1, c2 = st.columns([3, 1])
    c1.info(st.session_state.data_range_info)
    if st.session_state.last_update: c2.caption(f"更新: {st.session_state.last_update.strftime('%H:%M:%S')}")

    strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, st.session_state.df_view)
    df_display, entry_idx = strat.run_analysis(u_pos, u_time)
    
    last = df_display.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("價格", f"{last['Close']:.0f}")
    m2.metric("策略", last['Strategy_Action'])
    m3.metric("信心", last['Strategy_Detail'].split('(')[-1].replace(')', ''))

    df_chart = df_display.copy()
    df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
    total_len = len(df_chart)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['UB'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['LB'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', name='BB'))
    fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
    
    for act, sym, col, nm in [('買進', 'triangle-up', 'red', 'Buy'), ('放空', 'triangle-down', 'green', 'Sell'), ('出', 'x', 'gray', 'Exit')]:
        mask = df_chart['Strategy_Action'].str.contains(act)
        if mask.any():
            sub = df_chart[mask]
            fig.add_trace(go.Scatter(x=sub['Time'].dt.strftime('%H:%M'), y=sub['Close'], mode='markers', marker=dict(symbol=sym, size=12, color=col), name=nm))

    if entry_idx != -1 and entry_idx in df_chart.index:
        r = df_chart.loc[entry_idx]
        fig.add_trace(go.Scatter(x=[r['Time_Str']], y=[r['Close']], mode='markers', marker=dict(symbol='star', size=18, color='gold', line=dict(width=1, color='black')), name='My Entry'))

    fig.update_layout(height=550, margin=dict(t=30,l=10,r=10,b=10), xaxis=dict(type='category', rangeslider=dict(visible=True), range=[max(0, total_len-150), total_len-1]), legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📜 訊號履歷")
    st.dataframe(df_display.iloc[::-1], height=400, use_container_width=True, hide_index=True)
    
elif models is None: st.warning("⚠️ 缺少模型檔案")
else: st.info("👈 請點擊左側更新按鈕")
