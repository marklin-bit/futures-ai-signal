import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta, time as dt_time

# [Added] 自動刷新套件 (若無安裝會跳過)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# [Added] 引入 PyGithub 用於寫回檔案
try:
    from github import Github, Auth
except ImportError:
    st.error("請在 requirements.txt 中加入 'PyGithub'")

# ==========================================
# 1. 網頁設定與全域參數
# ==========================================
st.set_page_config(page_title="AI 交易訊號戰情室 (Pro)", layout="wide", initial_sidebar_state="expanded")

# CSS 美化
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 5rem;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
        .stButton button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# 2026 年月結算日清單 (維持手動更新以確保準確，避開台灣特殊假日)
SETTLEMENT_DATES_2026 = {
    '2026-01-21', '2026-02-18', '2026-03-18', '2026-04-15', '2026-05-20', '2026-06-17',
    '2026-07-15', '2026-08-19', '2026-09-16', '2026-10-21', '2026-11-18', '2026-12-16'
}

HIST_FILE_DAY = 'history_data_day.csv'
HIST_FILE_FULL = 'history_data_full.csv'

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
        """從鉅亨網抓取當日 5分K (含時間校正)"""
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        to_ts = int(datetime.now().timestamp())
        
        # 抓取最近 300 筆 (API 上限通常不長，所以依賴歷史檔拼接)
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 300}
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://stock.cnyes.com/market/{symbol}"
        }
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=5)
            data = res.json().get('data', {})
            
            if data.get('s') == 'ok' and data.get('t'):
                df = pd.DataFrame({
                    'Time': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
                })
                # 時區轉換與時間校正 (+5分鐘，標示 K 棒結束時間)
                df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
                df['Time'] = df['Time'] + timedelta(minutes=5)
                
                cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                return df
        except Exception as e:
            st.error(f"API 連線錯誤: {e}")
        return pd.DataFrame()

    def filter_day_session(self, df):
        """過濾日盤 (08:50 ~ 13:45)"""
        if df.empty: return df
        df = df.set_index('Time').sort_index()
        # 08:50 (第一根 08:45-08:50) 到 13:45 (最後一根)
        df_day = df.between_time(dt_time(8, 50), dt_time(13, 45)).reset_index()
        return df_day

    def calculate_indicators(self, df, mode='day'):
        """
        [優化] 使用向量化運算取代 for 迴圈，大幅提升效能。
        注意：傳入的 df 應包含「歷史資料 + 今日資料」，以確保指標運算有足夠長度。
        """
        if df.empty: return df
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']
        H = df['High']
        L = df['Low']
        O = df['Open']
        V = df['Volume']
        
        # 1. 布林通道 (20, 2)
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        df['UB'] = ma20 + 2 * std20 # 儲存供繪圖用
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        # 2. MA斜率
        ma_diff = ma20.diff()
        df['MA_Slope'] = np.sign(ma_diff).fillna(0) 
        
        # 3. 布林頻寬變化率
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        
        # 4. 相對成交量
        vol_ma5 = V.rolling(5).mean()
        df['Rel_Volume'] = V / vol_ma5
        
        # 5 & 6. KD (36, 3) - [優化] 向量化計算
        # KD 需要足夠的歷史資料來平滑 (建議至少 36*3 = 108 根)
        rsv_window = 36
        l_min = L.rolling(rsv_window).min()
        h_max = H.rolling(rsv_window).max()
        rsv = (C - l_min) / (h_max - l_min) # 0.0 ~ 1.0
        
        # 使用 Pandas EWM 模擬遞迴計算: alpha=1/3 等同於 (2/3)*Old + (1/3)*New
        # adjust=False 是關鍵，讓權重以前值為準
        df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
        
        # 7. 通道位置
        df['Position_in_Channel'] = (C - df['LB']) / (df['Bandwidth'].replace(0, np.nan))
        
        # 8. 波動率
        df['Volatility'] = (H - L) / C * 100
        
        # 9. K棒強度
        df['K_Strength'] = (C - O) / O * 100
        
        # 10. 實體佔比
        hl_range = (H - L).replace(0, 1)
        df['Body_Ratio'] = (C - O).abs() / hl_range
        
        # 11. 星期
        df['Week'] = df['Time'].dt.weekday + 1
        
        # 12 & 13. 結算日與時段
        if mode == 'full':
            df['Settlement_Day'] = 0
            df['Time_Segment'] = 1
        else:
            # 向量化處理日期判斷會比 apply 快，但在這裡資料量不大，apply 可接受
            def get_settlement(row):
                score = 0
                d = row['Time'].date()
                if d.weekday() in [2, 4]: score += 1 # 週三週五
                if str(d) in SETTLEMENT_DATES_2026: score += 1
                return score
            
            df['Settlement_Day'] = df.apply(get_settlement, axis=1)
            
            # 簡單時段劃分
            hours = df['Time'].dt.hour
            minutes = df['Time'].dt.minute
            hm = hours * 100 + minutes
            
            # 使用 numpy select 進行向量化條件判斷
            conditions = [
                hm <= 930,
                hm <= 1200
            ]
            choices = [0, 1]
            df['Time_Segment'] = np.select(conditions, choices, default=2)
        
        return df.fillna(0)

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
        # 尋找最近的符合時間
        mask = self.df['Time'].astype(str).str.contains(time_str, na=False)
        matches = self.df[mask]
        if not matches.empty:
            idx = matches.index[-1] 
            price = matches.loc[idx, 'Close']
            return idx, price
        return -1, 0.0

    def run_analysis(self, user_pos_type, entry_time_obj):
        if self.df.empty: return pd.DataFrame(), {}
        
        history_records = []
        X_all = self.df[self.processor.feature_cols]
        
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        user_entry_idx, user_cost = self.find_entry_info(entry_time_obj) if u_pos != "Empty" else (-1, 0.0)
        
        s_pos, s_price, s_idx = 0, 0.0, 0
        
        # 這裡依然使用 Loop，因為策略涉及持倉狀態 (Stateful)，向量化較困難
        # 但因為只跑「顯示用」的資料 (例如當天)，筆數不多，效能沒問題
        for i in range(len(self.df)):
            curr_time = self.df.iloc[i]['Time']
            curr_close = self.df.iloc[i]['Close']
            curr_feats = X_all.iloc[[i]]
            
            try:
                p_long = self.models['Long_Entry_Model'].predict_proba(curr_feats)[0][1]
                p_short = self.models['Short_Entry_Model'].predict_proba(curr_feats)[0][1]
            except Exception as e:
                # 若發生錯誤，填入預設值，避免當機
                p_long, p_short = 0.0, 0.0
            
            trend = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            
            # --- 策略模擬邏輯 (維持原樣) ---
            s_action = "⚪ 觀望"
            s_detail = trend
            
            if s_pos == 0:
                if p_long > self.params['entry'] and p_long > p_short:
                    s_pos, s_price, s_idx = 1, curr_close, i
                    s_action, s_detail = "🔴 買進", f"多 {p_long:.0%} {trend}"
                elif p_short > self.params['entry'] and p_short > p_long:
                    s_pos, s_price, s_idx = -1, curr_close, i
                    s_action, s_detail = "🟢 放空", f"空 {p_short:.0%} {trend}"
            elif s_pos == 1:
                pnl = curr_close - s_price
                if pnl <= -self.params['stop']:
                    s_pos, s_action, s_detail = 0, "💥 停損", f"損 {pnl:.0f}"
                else:
                    curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=i-s_idx)
                    exit_prob = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                    if exit_prob > self.params['exit']:
                        s_pos, s_action, s_detail = 0, "❌ 多出", f"帳{pnl:.0f}(出:{exit_prob:.0%})"
                    else:
                        s_action, s_detail = "⏳ 續抱", f"帳{pnl:.0f}(續:{1-exit_prob:.0%})"
            elif s_pos == -1:
                pnl = s_price - curr_close
                if pnl <= -self.params['stop']:
                    s_pos, s_action, s_detail = 0, "💥 停損", f"損 {pnl:.0f}"
                else:
                    curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=i-s_idx)
                    exit_prob = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                    if exit_prob > self.params['exit']:
                        s_pos, s_action, s_detail = 0, "❎ 空出", f"帳{pnl:.0f}(出:{exit_prob:.0%})"
                    else:
                        s_action, s_detail = "⏳ 續抱", f"帳{pnl:.0f}(續:{1-exit_prob:.0%})"

            # --- 真實部位建議邏輯 (維持原樣) ---
            u_action, u_note = "-", "-"
            
            if u_pos == "Empty":
                u_action, u_note = "未持單", "-"
            elif i < user_entry_idx:
                u_action, u_note = "未持單", "-"
            elif i == user_entry_idx:
                u_action = "🔴 多單進場" if u_pos == "Long" else "🟢 空單進場"
                u_note = f"成本 {user_cost:.0f}"
            else:
                hold_bars = i - user_entry_idx
                if u_pos == "Long":
                    pnl = curr_close - user_cost
                    if pnl <= -self.params['stop']:
                        u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        
                        detail_exit = f"帳面{pnl:.0f}(出:{ep:.0%}{trend})"
                        detail_hold = f"帳面{pnl:.0f}(續:{1-ep:.0%}{trend})"
                        
                        if ep > self.params['exit']:
                            u_action, u_note = "❌ 出場", detail_exit
                        elif p_long > self.params['entry'] and p_long > p_short:
                            u_action, u_note = "🔥 加碼", detail_hold
                        else:
                            u_action, u_note = "⏳ 續抱", detail_hold
                elif u_pos == "Short":
                    pnl = user_cost - curr_close
                    if pnl <= -self.params['stop']:
                        u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        
                        detail_exit = f"帳面{pnl:.0f}(出:{ep:.0%}{trend})"
                        detail_hold = f"帳面{pnl:.0f}(續:{1-ep:.0%}{trend})"
                        
                        if ep > self.params['exit']:
                            u_action, u_note = "❎ 出場", detail_exit
                        elif p_short > self.params['entry'] and p_short > p_long:
                            u_action, u_note = "🔥 加碼", detail_hold
                        else:
                            u_action, u_note = "⏳ 續抱", detail_hold

            history_records.append({
                'Time': curr_time, 'Close': curr_close,
                'UB': self.df.iloc[i].get('UB', 0), 'LB': self.df.iloc[i].get('LB', 0), # 增加布林軌道
                'Strategy_Action': s_action, 'Strategy_Detail': s_detail,
                'User_Advice': u_action, 'User_Note': u_note,
                'K': curr_feats['K'].values[0], 'D': curr_feats['D'].values[0], 
                'MA_Slope': curr_feats['MA_Slope'].values[0], 'Time_Segment': curr_feats['Time_Segment'].values[0],
                'Settlement_Day': curr_feats['Settlement_Day'].values[0]
            })
            
        return pd.DataFrame(history_records), user_entry_idx

# ==========================================
# 4. GitHub 存檔功能
# ==========================================
def push_to_github(file_path, df_to_save):
    token = st.secrets.get("GITHUB_TOKEN")
    repo_name = st.secrets.get("GITHUB_REPO")
    
    if not token or not repo_name:
        return "❌ 設定缺失: 請在 Secrets 設定 GITHUB_TOKEN 與 GITHUB_REPO"
        
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
    except Exception as e:
        return f"❌ GitHub 推送失敗: {e}"

# ==========================================
# 5. Streamlit UI
# ==========================================
@st.cache_resource
def load_models():
    try:
        paths = ['', 'models/']
        models = {}
        # 確保檔名與使用者提供的一致
        for name in ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']:
            for p in paths:
                if os.path.exists(f"{p}{name}.pkl"):
                    models[name] = joblib.load(f"{p}{name}.pkl"); break
        return models if len(models)==4 else None
    except: return None

# --- Layout ---
# 自動刷新: 每 5 分鐘 (300,000 毫秒) 刷新一次
if st_autorefresh:
    st_autorefresh(interval=300000, limit=None, key="auto_refresh")

left, right = st.columns([1, 2.5])
engine = DataEngine()
models = load_models()

with left:
    st.subheader("🛠️ 設定與資料")
    
    with st.expander("⚙️ 策略參數", expanded=False):
        p_entry = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        p_exit = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        p_stop = st.number_input("硬停損", 100, step=10)
    
    st.markdown("##### 👤 真實部位")
    u_pos = st.radio("持倉", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"], label_visibility="collapsed")
    u_time = None
    if u_pos != "空手 (Empty)":
        u_time = st.time_input("買進時間", value=dt_time(9,0), step=300)

    st.markdown("---")
    
    tab_r_day, tab_h_day, tab_r_full, tab_h_full = st.tabs(["🌞 即時(日)", "💾 歷史(日)", "🌙 即時(全)", "💾 歷史(全)"])
    
    # 處理資料的通用函數
    def process_data_pipeline(hist_file, mode):
        # 1. 讀取歷史資料
        df_hist = pd.read_csv(hist_file) if os.path.exists(hist_file) else pd.DataFrame()
        if not df_hist.empty: 
            df_hist['Time'] = pd.to_datetime(df_hist['Time'])
        
        # 2. 抓取 API 資料
        df_real = engine.fetch_realtime_from_anue()
        
        # 3. 合併與去重 (關鍵步驟：確保有足夠的歷史資料來算 KD)
        if not df_real.empty:
            df_total = pd.concat([df_hist, df_real])
            df_total = df_total.drop_duplicates(subset='Time', keep='last').sort_values('Time')
        else:
            df_total = df_hist
            
        if df_total.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 4. 根據模式過濾
        if mode == 'day':
            df_calc = engine.filter_day_session(df_total)
        else:
            df_calc = df_total

        # 5. 計算指標 (對整份資料計算，確保開盤時也有指標)
        df_calc = engine.calculate_indicators(df_calc, mode=mode)
        
        return df_calc, df_total

    # 1. 即時串接 (日)
    df_view = pd.DataFrame()
    entry_idx = -1
    
    with tab_r_day:
        st.caption("日盤模式：濾除夜盤，指標延續。")
        col_btn, col_info = st.columns([1, 1])
        if col_btn.button("🔄 立即更新", type="primary", key="btn_real_day"):
            with st.spinner("整合歷史與即時資料中..."):
                df_calc, _ = process_data_pipeline(HIST_FILE_DAY, 'day')
                
                if not df_calc.empty:
                    # 只取「今日」的資料來顯示，但指標是基於歷史算出來的
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    # 測試用: 若今日無資料(休市)，可暫時顯示最後一天
                    if df_calc['Time'].dt.strftime('%Y-%m-%d').eq(today_str).any():
                        df_view_raw = df_calc[df_calc['Time'].dt.strftime('%Y-%m-%d') == today_str]
                    else:
                        st.warning("今日尚無資料，顯示最近交易日")
                        last_date = df_calc['Time'].dt.date.iloc[-1]
                        df_view_raw = df_calc[df_calc['Time'].dt.date == last_date]
                    
                    if models:
                        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_view_raw)
                        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
                        st.success(f"更新成功！時間: {datetime.now().strftime('%H:%M:%S')}")
                    else:
                        st.error("❌ 找不到模型檔案 (.pkl)")
                else:
                    st.error("無資料可計算")

    # 2. 歷史管理 (日)
    with tab_h_day:
        st.caption("管理日盤歷史資料庫 (用於計算開盤指標)")
        up_day = st.file_uploader("上傳歷史檔", type=['csv'], key="up_day")
        if up_day:
            pd.read_csv(up_day).to_csv(HIST_FILE_DAY, index=False)
            st.success("已更新本地歷史檔")
        
        if st.button("☁️ 寫入 GitHub (日盤)", key="save_day"):
            # 重新執行一次 Pipeline 確保存入的是最新合併結果
            _, df_total_day = process_data_pipeline(HIST_FILE_DAY, 'day')
            if not df_total_day.empty:
                # 只存原始資料，不存指標
                save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                # 重新過濾一次日盤 (因為 process_data_pipeline 回傳的 df_total 是含夜盤的原始拼接)
                df_to_save = engine.filter_day_session(df_total_day)[save_cols]
                
                df_to_save.to_csv(HIST_FILE_DAY, index=False)
                if "GITHUB_TOKEN" in st.secrets:
                    with st.spinner("推送到 GitHub..."):
                        st.write(push_to_github(HIST_FILE_DAY, df_to_save))
                else:
                    st.error("請設定 GITHUB_TOKEN")

    # 3. 即時串接 (全)
    with tab_r_full:
        st.caption("全盤模式：包含夜盤。")
        if st.button("🔄 立即更新", key="btn_real_full"):
             with st.spinner("計算中..."):
                df_calc, _ = process_data_pipeline(HIST_FILE_FULL, 'full')
                if not df_calc.empty:
                    df_view_raw = df_calc.tail(300) # 全盤顯示最後 300 筆
                    if models:
                        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_view_raw)
                        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
                        st.success("更新成功")

    # 4. 歷史管理 (全)
    with tab_h_full:
        st.caption("管理全盤歷史資料庫")
        up_full = st.file_uploader("上傳歷史檔", type=['csv'], key="up_full")
        if up_full:
            pd.read_csv(up_full).to_csv(HIST_FILE_FULL, index=False)
            st.success("已更新")
            
        if st.button("☁️ 寫入 GitHub (全盤)", key="save_full"):
            _, df_total_full = process_data_pipeline(HIST_FILE_FULL, 'full')
            if not df_total_full.empty:
                save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                df_to_save = df_total_full[save_cols]
                df_to_save.to_csv(HIST_FILE_FULL, index=False)
                if "GITHUB_TOKEN" in st.secrets:
                    with st.spinner("推送到 GitHub..."):
                        st.write(push_to_github(HIST_FILE_FULL, df_to_save))

with right:
    if not df_view.empty:
        st.subheader("📊 戰情走勢圖")
        
        # 準備繪圖資料
        df_chart = df_view.copy()
        df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
        
        fig = go.Figure()
        
        # [視覺化增強] 1. 布林通道 (帶狀)
        # 上軌
        fig.add_trace(go.Scatter(
            x=df_chart['Time_Str'], y=df_chart['UB'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        # 下軌 (填滿到上軌)
        fig.add_trace(go.Scatter(
            x=df_chart['Time_Str'], y=df_chart['LB'],
            mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor='rgba(173, 216, 230, 0.2)', # 淺藍色半透明
            name='Bollinger Bands'
        ))
        
        # 2. K線收盤價
        fig.add_trace(go.Scatter(
            x=df_chart['Time_Str'], y=df_chart['Close'], 
            mode='lines', name='Price', line=dict(color='#1f77b4', width=2)
        ))
        
        # 3. 交易訊號標記
        buys = df_chart[df_chart['Strategy_Action'].str.contains('買進')]
        sells = df_chart[df_chart['Strategy_Action'].str.contains('放空')]
        exits_long = df_chart[df_chart['Strategy_Action'].str.contains('❌')]
        exits_short = df_chart[df_chart['Strategy_Action'].str.contains('❎')]
        
        if not buys.empty: 
            fig.add_trace(go.Scatter(x=buys['Time'].dt.strftime('%H:%M'), y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='Buy'))
        if not sells.empty: 
            fig.add_trace(go.Scatter(x=sells['Time'].dt.strftime('%H:%M'), y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='Sell'))
        if not exits_long.empty: 
            fig.add_trace(go.Scatter(x=exits_long['Time'].dt.strftime('%H:%M'), y=exits_long['Close'], mode='markers', marker=dict(symbol='x', size=10, color='red'), name='Long Exit'))
        if not exits_short.empty: 
            fig.add_trace(go.Scatter(x=exits_short['Time'].dt.strftime('%H:%M'), y=exits_short['Close'], mode='markers', marker=dict(symbol='x', size=10, color='green'), name='Short Exit'))
        
        # 4. 使用者進場點
        if entry_idx != -1 and entry_idx in df_chart.index:
            entry_row = df_chart.loc[entry_idx]
            color = 'red' if u_pos == "多單 (Long)" else 'green'
            fig.add_trace(go.Scatter(
                x=[entry_row['Time_Str']], y=[entry_row['Close']], 
                mode='markers', marker=dict(symbol='star', size=18, color=color, line=dict(width=2, color='gold')), 
                name='My Entry'
            ))

        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0), 
            height=400, 
            xaxis_type='category',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📜 訊號履歷")
        st.dataframe(
            df_chart.iloc[::-1], # 倒序顯示
            height=300,
            column_config={
                "Time": st.column_config.DatetimeColumn("時間", format="HH:mm", width="small"),
                "Close": st.column_config.NumberColumn("價位", format="%d", width="small"),
                "Strategy_Action": st.column_config.TextColumn("策略", width="small"),
                "Strategy_Detail": st.column_config.TextColumn("多空機率", width="medium"),
                "User_Advice": st.column_config.TextColumn("建議", width="small"),
                "UB": None, "LB": None, # 隱藏通道數據
                "K": None, "D": None, "MA_Slope": None, "Time_Segment": None, "Settlement_Day": None 
            },
            use_container_width=True,
            hide_index=True
        )
        
    elif models is None:
        st.error("⚠️ 模型載入失敗: 請確認 models/ 資料夾下是否有 4 個 .pkl 檔案")
    else:
        st.info("👈 請點擊左側「立即更新」開始載入資料")
