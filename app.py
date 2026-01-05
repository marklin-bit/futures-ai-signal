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

# 定義兩個獨立的資料庫檔案
HIST_FILE_DAY = 'history_data_day.csv'   # 純日盤資料庫
HIST_FILE_FULL = 'history_data_full.csv' # 全盤資料庫

# Session State 初始化
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
        """解析鉅亨網 API"""
        if not data.get('t'): return pd.DataFrame()
        df = pd.DataFrame({
            'Time': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
        })
        # UTC -> Taiwan -> +5min (K棒結束時間)
        df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
        df['Time'] = df['Time'] + timedelta(minutes=5)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
        return df

    def fetch_anue_raw(self):
        """抓取 API 新資料"""
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://stock.cnyes.com/market/{symbol}"}
        
        to_ts = int(datetime.now().timestamp())
        # 抓取 1000 筆 (約 3-4 天)
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 1000}
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=8)
            data = res.json().get('data', {})
            if data.get('t'):
                return self._parse_anue_response(data)
        except Exception as e:
            st.error(f"鉅亨網連線錯誤: {e}")
        
        return pd.DataFrame()

    def merge_and_save(self, api_df, hist_file, is_day_mode=False):
        """
        [增強版] 合併、過濾、存檔、並自動清理過期資料 (只留最近 5 個交易日)
        """
        # 1. 讀取歷史
        if os.path.exists(hist_file):
            try:
                hist_df = pd.read_csv(hist_file)
                hist_df['Time'] = pd.to_datetime(hist_df['Time'])
            except:
                hist_df = pd.DataFrame()
        else:
            hist_df = pd.DataFrame()

        # 2. 處理新資料
        new_df = api_df.copy()
        if not new_df.empty and is_day_mode:
            # 日盤模式：嚴格過濾，只保留 08:45 ~ 13:45 的資料
            new_df = new_df.set_index('Time').sort_index()
            new_df = new_df.between_time(dt_time(8, 45), dt_time(13, 45)).reset_index()

        # 3. 合併與去重
        if not new_df.empty:
            if not hist_df.empty:
                full_df = pd.concat([hist_df, new_df])
            else:
                full_df = new_df
            
            # 依時間去重，保留最新的數據
            full_df = full_df.drop_duplicates(subset='Time', keep='last').sort_values('Time').reset_index(drop=True)
        else:
            full_df = hist_df

        # 確保日盤歷史檔不含雜質
        if is_day_mode and not full_df.empty:
             full_df = full_df.set_index('Time').sort_index()
             full_df = full_df.between_time(dt_time(8, 45), dt_time(13, 45)).reset_index()

        # 4. 自動清理：只保留最近 5 個交易日
        if not full_df.empty:
            unique_dates = full_df['Time'].dt.date.unique()
            unique_dates.sort()
            
            if len(unique_dates) > 5:
                cutoff_date = unique_dates[-5]
                full_df = full_df[full_df['Time'].dt.date >= cutoff_date]

        # 5. 存檔
        if not full_df.empty:
            save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            full_df[save_cols].to_csv(hist_file, index=False)
            
            start_str = full_df['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')
            end_str = full_df['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M')
            st.session_state.data_range_info = f"{start_str} ~ {end_str} (共 {len(full_df)} 筆 / {len(unique_dates) if 'unique_dates' in locals() else '?'} 天)"
        else:
            st.session_state.data_range_info = "無資料"

        return full_df

    def calculate_indicators(self, df, mode='day'):
        """計算技術指標"""
        if df.empty: return df
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']; H = df['High']; L = df['Low']; O = df['Open']; V = df['Volume']
        
        # 1. 布林通道 (20, 2)
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        df['UB'] = ma20 + 2 * std20
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        # 2. 其他特徵
        df['MA_Slope'] = np.sign(ma20.diff()) 
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        df['Rel_Volume'] = V / V.rolling(5).mean()
        
        # 3. KD (36, 3)
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
# 3. 策略引擎 (無變更)
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
    
    if not token or not repo_name:
        return "❌ 缺少 GitHub 設定"
    
    # Repo 格式檢查
    if "/" not in repo_name:
        return f"❌ Repo 名稱錯誤: '{repo_name}'。請務必使用 'username/repo_name' 格式！"

    try:
        g = Github(token)
        # 這裡會觸發 404 如果 Token 權限不夠或 Repo 不存在
        repo = g.get_repo(repo_name)
        
        csv_content = df_to_save.to_csv(index=False)
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, f"Update {file_path}", csv_content, contents.sha)
            return "✅ 雲端更新成功！"
        except:
            repo.create_file(file_path, f"Create {file_path}", csv_content)
            return "✅ 雲端建立成功！"
    except Exception as e:
        # [Fix] 更詳細的錯誤提示
        err_msg = str(e)
        if "404" in err_msg and "Not Found" in err_msg:
            return (
                f"❌ GitHub 回傳 404 錯誤 (找不到 Repo)。請檢查：\n"
                f"1. Token 是否已勾選 'repo' (Full control) 權限？(私有庫必須)\n"
                f"2. Repo 名稱 '{repo_name}' 是否完全正確？\n"
                f"3. 該 Repo 是否真的存在？"
            )
        return f"❌ GitHub 推送失敗: {e}"

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

    with st.expander("💾 資料庫管理", expanded=False):
        st.caption("手動上傳/下載 CSV 備份")
        
        tab_db_day, tab_db_full = st.tabs(["日盤庫", "全盤庫"])
        
        with tab_db_day:
            up_day = st.file_uploader("上傳日盤歷史", type=['csv'], key="up_day")
            if up_day:
                pd.read_csv(up_day).to_csv(HIST_FILE_DAY, index=False)
                st.success("已更新日盤庫")
            if st.button("寫入 GitHub (日盤)", key="git_day"):
                if os.path.exists(HIST_FILE_DAY):
                    st.write(push_to_github(HIST_FILE_DAY, pd.read_csv(HIST_FILE_DAY)))
                else: st.error("無本地檔")

        with tab_db_full:
            up_full = st.file_uploader("上傳全盤歷史", type=['csv'], key="up_full")
            if up_full:
                pd.read_csv(up_full).to_csv(HIST_FILE_FULL, index=False)
                st.success("已更新全盤庫")
            if st.button("寫入 GitHub (全盤)", key="git_full"):
                if os.path.exists(HIST_FILE_FULL):
                    st.write(push_to_github(HIST_FILE_FULL, pd.read_csv(HIST_FILE_FULL)))
                else: st.error("無本地檔")

def process_data(mode):
    # 1. 判斷要用的歷史檔
    hist_file = HIST_FILE_DAY if mode == 'day' else HIST_FILE_FULL
    
    # 2. 抓取 API 新資料 (鉅亨網)
    api_df = engine.fetch_anue_raw()
    
    # 3. 讀取與合併
    # 注意: merge_and_save 裡面會負責日盤過濾 & 自動清理
    final_df = engine.merge_and_save(api_df, hist_file, is_day_mode=(mode=='day'))
    
    if final_df.empty:
        return pd.DataFrame(), "無資料 (API 失敗且無歷史檔)"
        
    # 如果是日盤模式，但 API 沒給東西 (表示收盤了)，要特別標示
    status = "OK"
    if api_df.empty:
        status = "⚠️ API 無新資料，僅顯示歷史存檔 (可能已收盤)"
    
    # 4. 計算指標
    df_calc = engine.calculate_indicators(final_df, mode=mode)
    
    return df_calc, status

if trigger_day:
    with st.spinner("整合日盤數據中..."):
        df_res, status = process_data('day')
        
        if not df_res.empty:
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'day'
            st.session_state.last_update = datetime.now()
            
            # [Fix] 如果不是 OK，就 toast 警告一下，不要顯示綠色成功
            if status != "OK":
                st.toast(status, icon="⚠️")
        else:
            st.error(status)

if trigger_full:
    with st.spinner("整合全盤數據中..."):
        df_res, status = process_data('full')
        if not df_res.empty:
            st.session_state.df_view = df_res
            st.session_state.current_mode = 'full'
            st.session_state.last_update = datetime.now()
            if status != "OK":
                st.toast(status, icon="⚠️")
        else:
            st.error(status)

if not st.session_state.df_view.empty and models:
    mode_name = "🌞 日盤" if st.session_state.current_mode == 'day' else "🌙 全盤"
    st.title(f"{mode_name}戰情室")
    
    # 顯示資料庫狀態
    if st.session_state.data_range_info:
        st.info(f"💾 資料庫範圍 (最近 5 日): {st.session_state.data_range_info}")
    
    st.caption(f"最後更新: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    if len(st.session_state.df_view) < 50:
        st.warning(f"⚠️ 資料筆數 ({len(st.session_state.df_view)}) 不足，技術指標可能偏差。")

    strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, st.session_state.df_view)
    df_display, entry_idx = strat.run_analysis(u_pos,
