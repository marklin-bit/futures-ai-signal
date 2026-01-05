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
    pass # 錯誤提示移到下方顯示，避免中斷

# ==========================================
# 1. 網頁設定與全域參數
# ==========================================
st.set_page_config(page_title="AI 交易訊號戰情室 (Pro)", layout="wide", initial_sidebar_state="expanded")

# [Fix] CSS 美化修復：解決文字裁切與版面擁擠問題
st.markdown("""
    <style>
        /* 增加頂部間距，避免標題被瀏覽器或 Streamlit 頂部 Bar 遮擋 */
        .block-container {
            padding-top: 3.5rem !important; 
            padding-bottom: 5rem;
            max-width: 95% !important; /* 讓寬螢幕更舒適 */
        }
        /* 調整側邊欄寬度與間距 */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 3rem;
        }
        div[data-testid="stMetricValue"] {font-size: 24px;}
        .stButton button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# 2026 年月結算日清單
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
        """從鉅亨網抓取 5分K"""
        symbol = "TWF:TXF:FUTURES"
        url = "https://ws.api.cnyes.com/ws/api/v1/charting/history"
        to_ts = int(datetime.now().timestamp())
        
        # [Fix] 將 limit 提高到 1000，確保即使歷史檔有缺漏，也能抓回最近 3-5 天資料來算指標
        # 這樣明天開盤即使沒有今天的 CSV，也能靠 API 補足指標計算所需的暖機長度
        params = {"symbol": symbol, "resolution": "5", "to": to_ts, "limit": 1000}
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://stock.cnyes.com/market/{symbol}"
        }
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=8)
            data = res.json().get('data', {})
            
            if data.get('s') == 'ok' and data.get('t'):
                df = pd.DataFrame({
                    'Time': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']
                })
                # 時區轉換 (+5分鐘校正)
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
        df_day = df.between_time(dt_time(8, 50), dt_time(13, 45)).reset_index()
        return df_day

    def calculate_indicators(self, df, mode='day'):
        if df.empty: return df
        df = df.sort_values('Time').reset_index(drop=True)
        
        C = df['Close']; H = df['High']; L = df['Low']; O = df['Open']; V = df['Volume']
        
        # 1. 布林通道 (20, 2)
        ma20 = C.rolling(20).mean()
        std20 = C.rolling(20).std()
        df['UB'] = ma20 + 2 * std20
        df['LB'] = ma20 - 2 * std20
        df['Bandwidth'] = df['UB'] - df['LB']
        
        # 2. MA斜率
        df['MA_Slope'] = np.sign(ma20.diff()).fillna(0) 
        
        # 3. 布林頻寬變化率
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        
        # 4. 相對成交量
        vol_ma5 = V.rolling(5).mean()
        df['Rel_Volume'] = V / vol_ma5
        
        # 5 & 6. KD (36, 3) - 向量化
        rsv_window = 36
        l_min = L.rolling(rsv_window).min()
        h_max = H.rolling(rsv_window).max()
        rsv = (C - l_min) / (h_max - l_min)
        
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
            def get_settlement(row):
                score = 0
                d = row['Time'].date()
                if d.weekday() in [2, 4]: score += 1
                if str(d) in SETTLEMENT_DATES_2026: score += 1
                return score
            df['Settlement_Day'] = df.apply(get_settlement, axis=1)
            
            hours = df['Time'].dt.hour
            minutes = df['Time'].dt.minute
            hm = hours * 100 + minutes
            conditions = [hm <= 930, hm <= 1200]
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
        
        for i in range(len(self.df)):
            curr_time = self.df.iloc[i]['Time']
            curr_close = self.df.iloc[i]['Close']
            curr_feats = X_all.iloc[[i]]
            
            try:
                p_long = self.models['Long_Entry_Model'].predict_proba(curr_feats)[0][1]
                p_short = self.models['Short_Entry_Model'].predict_proba(curr_feats)[0][1]
            except: p_long, p_short = 0.0, 0.0
            
            trend = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            s_action, s_detail = "⚪ 觀望", trend
            
            # --- 簡化後的策略邏輯 ---
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

            # --- 真實部位建議邏輯 ---
            u_action, u_note = "-", "-"
            if u_pos != "Empty" and i >= user_entry_idx:
                hold_bars = i - user_entry_idx
                if u_pos == "Long":
                    pnl = curr_close - user_cost
                    if i == user_entry_idx: u_action, u_note = "🔴 多單進場", f"成本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        detail = f"帳{pnl:.0f}(出:{ep:.0%})"
                        if ep > self.params['exit']: u_action, u_note = "❌ 出場", detail
                        elif p_long > self.params['entry']: u_action, u_note = "🔥 加碼", detail
                        else: u_action, u_note = "⏳ 續抱", detail
                elif u_pos == "Short":
                    pnl = user_cost - curr_close
                    if i == user_entry_idx: u_action, u_note = "🟢 空單進場", f"成本 {user_cost:.0f}"
                    elif pnl <= -self.params['stop']: u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        detail = f"帳{pnl:.0f}(出:{ep:.0%})"
                        if ep > self.params['exit']: u_action, u_note = "❎ 出場", detail
                        elif p_short > self.params['entry']: u_action, u_note = "🔥 加碼", detail
                        else: u_action, u_note = "⏳ 續抱", detail

            history_records.append({
                'Time': curr_time, 'Close': curr_close,
                'UB': self.df.iloc[i].get('UB', 0), 'LB': self.df.iloc[i].get('LB', 0),
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
    if not token or not repo_name: return "❌ 請設定 Secrets: GITHUB_TOKEN 與 GITHUB_REPO"
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
    except Exception as e: return f"❌ 推送失敗: {e}"

# ==========================================
# 5. Streamlit UI
# ==========================================
@st.cache_resource
def load_models():
    try:
        paths = ['', 'models/']
        models = {}
        for name in ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']:
            for p in paths:
                if os.path.exists(f"{p}{name}.pkl"):
                    models[name] = joblib.load(f"{p}{name}.pkl"); break
        return models if len(models)==4 else None
    except: return None

# 自動刷新
if st_autorefresh: st_autorefresh(interval=300000, limit=None, key="auto_refresh")

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
    if u_pos != "空手 (Empty)": u_time = st.time_input("買進時間", value=dt_time(9,0), step=300)
    st.markdown("---")
    
    tab_r_day, tab_h_day, tab_r_full, tab_h_full = st.tabs(["🌞 即時(日)", "💾 歷史(日)", "🌙 即時(全)", "💾 歷史(全)"])
    
    def process_data_pipeline(hist_file, mode):
        df_hist = pd.read_csv(hist_file) if os.path.exists(hist_file) else pd.DataFrame()
        if not df_hist.empty: df_hist['Time'] = pd.to_datetime(df_hist['Time'])
        
        # API 抓取 (1000筆)
        df_real = engine.fetch_realtime_from_anue()
        
        # 合併與去重
        if not df_real.empty:
            df_total = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time', keep='last').sort_values('Time')
        else:
            df_total = df_hist
            
        if df_total.empty: return pd.DataFrame(), pd.DataFrame(), "No Data (Both History and API empty)"

        # 模式過濾
        if mode == 'day':
            df_calc = engine.filter_day_session(df_total)
            if df_calc.empty: return pd.DataFrame(), df_total, "No Day Session Data (Filtered out)"
        else:
            df_calc = df_total

        df_calc = engine.calculate_indicators(df_calc, mode=mode)
        return df_calc, df_total, "OK"

    # 1. 即時 (日)
    df_view = pd.DataFrame()
    entry_idx = -1
    
    with tab_r_day:
        if st.button("🔄 立即更新", type="primary", key="btn_real_day"):
            with st.spinner("抓取最近 1000 筆資料計算中..."):
                df_calc, _, status = process_data_pipeline(HIST_FILE_DAY, 'day')
                
                if status == "OK" and not df_calc.empty:
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    # 優先顯示今日，若無今日則顯示最近一日
                    if df_calc['Time'].dt.strftime('%Y-%m-%d').eq(today_str).any():
                        df_view_raw = df_calc[df_calc['Time'].dt.strftime('%Y-%m-%d') == today_str]
                        st.caption(f"顯示資料: {today_str} (今日)")
                    else:
                        last_date = df_calc['Time'].dt.date.iloc[-1]
                        df_view_raw = df_calc[df_calc['Time'].dt.date == last_date]
                        st.warning(f"今日尚未開盤或無資料，顯示最近交易日: {last_date}")

                    if models:
                        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_view_raw)
                        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
                        st.success(f"更新成功！")
                    else: st.error("❌ 找不到模型檔案")
                else:
                    st.error(f"無資料可計算: {status}")
                    st.caption("建議: 1. 檢查 API 連線 2. 歷史檔是否上傳 3. 是否為夜盤時段(日盤模式下無資料為正常)")

    # 2. 歷史 (日)
    with tab_h_day:
        up_day = st.file_uploader("上傳歷史檔", type=['csv'], key="up_day")
        if up_day:
            pd.read_csv(up_day).to_csv(HIST_FILE_DAY, index=False)
            st.success("已更新")
        if st.button("☁️ 寫入 GitHub (日盤)", key="save_day"):
            _, df_total_day, _ = process_data_pipeline(HIST_FILE_DAY, 'day')
            if not df_total_day.empty:
                df_to_save = engine.filter_day_session(df_total_day)[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df_to_save.to_csv(HIST_FILE_DAY, index=False)
                if "GITHUB_TOKEN" in st.secrets:
                    with st.spinner("推送到 GitHub..."): st.write(push_to_github(HIST_FILE_DAY, df_to_save))
                else: st.error("請設定 Secrets")

    # 3. 即時 (全)
    with tab_r_full:
        if st.button("🔄 立即更新", key="btn_real_full"):
             with st.spinner("計算中..."):
                df_calc, _, status = process_data_pipeline(HIST_FILE_FULL, 'full')
                if status == "OK":
                    df_view_raw = df_calc.tail(300)
                    if models:
                        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_view_raw)
                        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
                        st.success("更新成功")

    # 4. 歷史 (全)
    with tab_h_full:
        up_full = st.file_uploader("上傳歷史檔", type=['csv'], key="up_full")
        if up_full:
            pd.read_csv(up_full).to_csv(HIST_FILE_FULL, index=False)
            st.success("已更新")
        if st.button("☁️ 寫入 GitHub (全盤)", key="save_full"):
            _, df_total_full, _ = process_data_pipeline(HIST_FILE_FULL, 'full')
            if not df_total_full.empty:
                df_to_save = df_total_full[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df_to_save.to_csv(HIST_FILE_FULL, index=False)
                if "GITHUB_TOKEN" in st.secrets:
                    with st.spinner("推送到 GitHub..."): st.write(push_to_github(HIST_FILE_FULL, df_to_save))

with right:
    if not df_view.empty:
        st.subheader("📊 戰情走勢圖")
        df_chart = df_view.copy()
        df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
        
        # [Fix] 預設只顯示最後 100 筆，避免擠成一團，但保留完整資料在物件中
        display_range = 100
        total_len = len(df_chart)
        start_idx = max(0, total_len - display_range)
        
        fig = go.Figure()
        
        # 上軌
        fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['UB'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # 下軌 (填色)
        fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['LB'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', name='BB'))
        # 收盤價
        fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
        
        # 訊號
        buys = df_chart[df_chart['Strategy_Action'].str.contains('買進')]
        sells = df_chart[df_chart['Strategy_Action'].str.contains('放空')]
        exits = df_chart[df_chart['Strategy_Action'].str.contains('出')]
        
        if not buys.empty: fig.add_trace(go.Scatter(x=buys['Time'].dt.strftime('%H:%M'), y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='Buy'))
        if not sells.empty: fig.add_trace(go.Scatter(x=sells['Time'].dt.strftime('%H:%M'), y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='Sell'))
        if not exits.empty: fig.add_trace(go.Scatter(x=exits['Time'].dt.strftime('%H:%M'), y=exits['Close'], mode='markers', marker=dict(symbol='x', size=10, color='gray'), name='Exit'))
        
        if entry_idx != -1 and entry_idx in df_chart.index:
            entry_row = df_chart.loc[entry_idx]
            fig.add_trace(go.Scatter(x=[entry_row['Time_Str']], y=[entry_row['Close']], mode='markers', marker=dict(symbol='star', size=18, color='gold', line=dict(width=1, color='black')), name='My Entry'))

        # [Fix] 介面優化：開啟範圍滑桿(rangeslider) 與 滑鼠縮放，並設定預設顯示範圍
        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0), 
            height=450, 
            xaxis=dict(
                type='category', 
                rangeslider=dict(visible=True), # 顯示縮放條
                range=[max(0, total_len - 100), total_len - 1] # 預設顯示最後 100 筆
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            dragmode="pan" # 預設為平移模式，方便查看
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📜 訊號履歷")
        st.dataframe(
            df_chart.iloc[::-1], 
            height=300,
            column_config={
                "Time": st.column_config.DatetimeColumn("時間", format="HH:mm", width="small"),
                "Close": st.column_config.NumberColumn("價位", format="%d", width="small"),
                "Strategy_Action": st.column_config.TextColumn("策略", width="small"),
                "Strategy_Detail": st.column_config.TextColumn("多空機率", width="medium"),
                "User_Advice": st.column_config.TextColumn("建議", width="small"),
                "UB": None, "LB": None
            },
            use_container_width=True,
            hide_index=True
        )
        
    elif models is None:
        st.error("⚠️ 模型載入失敗: 請確認 models/ 資料夾下是否有 4 個 .pkl 檔案")
    else:
        st.info("👈 請點擊左側「立即更新」")
