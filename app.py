import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta, time as dt_time
import io

# 1. 網頁設定
st.set_page_config(page_title="AI 交易訊號戰情室 (Pro)", layout="wide", initial_sidebar_state="expanded")

# CSS 美化
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 5rem;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
        .stButton button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# 2026 年月結算日清單 (預估為每月第三個週三)
SETTLEMENT_DATES_2026 = {
    '2026-01-21', '2026-02-18', '2026-03-18', '2026-04-15', '2026-05-20', '2026-06-17',
    '2026-07-15', '2026-08-19', '2026-09-16', '2026-10-21', '2026-11-18', '2026-12-16'
}

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
        
        # [Modify] 抓取量調回 300 (足夠涵蓋今日日盤 + 昨日夜盤)
        # 依靠使用者上傳的 history.csv 來提供足夠的計算基底
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
                # 時區轉換與時間校正 (+5分鐘: 開盤時間->收盤時間)
                df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
                df['Time'] = df['Time'] + timedelta(minutes=5)
                
                cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                return df
        except: pass
        return pd.DataFrame()

    def filter_day_session(self, df):
        """過濾日盤 (08:50 ~ 13:45)"""
        if df.empty: return df
        df = df.set_index('Time').sort_index()
        # 結算日可能 13:30 收，正常日 13:45
        df_day = df.between_time(dt_time(8, 50), dt_time(13, 45)).reset_index()
        return df_day

    def calculate_indicators(self, df):
        """
        依照使用者指定的公式計算 13 個特徵
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
        ub = ma20 + 2 * std20
        lb = ma20 - 2 * std20
        
        df['Bandwidth'] = ub - lb
        
        # 2. MA斜率 (MA_Slope): 正值1, 負值-1, 0為0
        # 邏輯: 當前MA - 前一次MA
        ma_diff = ma20.diff()
        df['MA_Slope'] = np.sign(ma_diff).fillna(0) 
        
        # 3. 布林頻寬變化率 (Bandwidth_Rate)
        # (當前BW - 前一次BW) / 前一次BW
        df['Bandwidth_Rate'] = df['Bandwidth'].pct_change()
        
        # 4. 相對成交量 (Rel_Volume) = V / 5MA_V
        vol_ma5 = V.rolling(5).mean()
        df['Rel_Volume'] = V / vol_ma5
        
        # 5 & 6. KD (36, 3) - 手動計算
        # [Checked] 移除 *100，讓數值維持在 0~1 (符合訓練資料格式)
        rsv_window = 36
        l_min = L.rolling(rsv_window).min()
        h_max = H.rolling(rsv_window).max()
        rsv = (C - l_min) / (h_max - l_min) # 0.0 ~ 1.0
        
        # EMA Smoothing for K and D (alpha=1/3)
        # [Modify] 初始值設為 0.5 (對應 50%)
        k_vals = [0.5] * len(df)
        d_vals = [0.5] * len(df)
        
        # 轉成 numpy 加速
        rsv_np = rsv.to_numpy()
        
        for i in range(1, len(df)):
            if np.isnan(rsv_np[i]): 
                k_vals[i] = k_vals[i-1]
                d_vals[i] = d_vals[i-1]
            else:
                k_vals[i] = (2/3) * k_vals[i-1] + (1/3) * rsv_np[i]
                d_vals[i] = (2/3) * d_vals[i-1] + (1/3) * k_vals[i]
                
        df['K'] = k_vals
        df['D'] = d_vals
        
        # 7. 通道位置
        df['Position_in_Channel'] = (C - lb) / (ub - lb)
        
        # 8. 波動率: (H-L)/C * 100
        df['Volatility'] = (H - L) / C * 100
        
        # 9. K棒強度: (C-O)/O * 100
        df['K_Strength'] = (C - O) / O * 100
        
        # 10. 實體佔比: ABS((C-O)/(H-L))
        hl_range = (H - L).replace(0, 1) # 防除以0
        df['Body_Ratio'] = (C - O).abs() / hl_range
        
        # 11. 星期 (1=Mon, ..., 5=Fri)
        df['Week'] = df['Time'].dt.weekday + 1
        
        # 12. 結算日 (Settlement_Day)
        def get_settlement(row):
            score = 0
            d = row['Time'].date()
            if d.weekday() in [2, 4]: # Wed(2) or Fri(4)
                score += 1
            if str(d) in SETTLEMENT_DATES_2026:
                score += 1
            return score
            
        df['Settlement_Day'] = df.apply(get_settlement, axis=1)
        
        # 13. 時段 (Time_Segment)
        def get_segment(t):
            hm = t.hour * 100 + t.minute
            if hm <= 930: return 0   # 08:50 - 09:30
            elif hm <= 1200: return 1 # 09:35 - 12:00
            else: return 2           # 12:05 後
            
        df['Time_Segment'] = df['Time'].apply(get_segment)
        
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
        
        # 使用者部位資訊
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        user_entry_idx, user_cost = self.find_entry_info(entry_time_obj) if u_pos != "Empty" else (-1, 0.0)
        
        # 策略模擬變數
        s_pos, s_price, s_idx = 0, 0.0, 0
        
        for i in range(len(self.df)):
            curr_time = self.df.iloc[i]['Time']
            curr_close = self.df.iloc[i]['Close']
            curr_feats = X_all.iloc[[i]]
            
            # 預測
            p_long = self.models['Long_Entry_Model'].predict_proba(curr_feats)[0][1]
            p_short = self.models['Short_Entry_Model'].predict_proba(curr_feats)[0][1]
            
            trend = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            
            # --- 1. 模型策略 (模擬) ---
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
                    exit_prob = self.models['Long_Exit_Model'].predict_proba(curr_feats[self.processor.exit_feature_cols].assign(Floating_PnL=pnl, Hold_Bars=i-s_idx))[0][1]
                    if exit_prob > self.params['exit']:
                        s_pos, s_action, s_detail = 0, "❌ 多出", f"帳{pnl:.0f}(出:{exit_prob:.0%})"
                    else:
                        s_action, s_detail = "⏳ 續抱", f"帳{pnl:.0f}(續:{1-exit_prob:.0%})"
            elif s_pos == -1:
                pnl = s_price - curr_close
                if pnl <= -self.params['stop']:
                    s_pos, s_action, s_detail = 0, "💥 停損", f"損 {pnl:.0f}"
                else:
                    exit_prob = self.models['Short_Exit_Model'].predict_proba(curr_feats[self.processor.exit_feature_cols].assign(Floating_PnL=pnl, Hold_Bars=i-s_idx))[0][1]
                    if exit_prob > self.params['exit']:
                        s_pos, s_action, s_detail = 0, "❎ 空出", f"帳{pnl:.0f}(出:{exit_prob:.0%})"
                    else:
                        s_action, s_detail = "⏳ 續抱", f"帳{pnl:.0f}(續:{1-exit_prob:.0%})"

            # --- 2. 持單建議 (真實) ---
            u_action, u_note = "-", "-"
            
            if u_pos == "Empty":
                u_action, u_note = "未持單", "-"
            elif i < user_entry_idx:
                u_action, u_note = "未持單", "-"
            elif i == user_entry_idx:
                u_action = "🔴 多單進場" if u_pos == "Long" else "🟢 空單進場"
                u_note = f"成本 {user_cost:.0f}"
            else: # 持倉
                hold_bars = i - user_entry_idx
                if u_pos == "Long":
                    pnl = curr_close - user_cost
                    if pnl <= -self.params['stop']:
                        u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        ep = self.models['Long_Exit_Model'].predict_proba(curr_feats[self.processor.exit_feature_cols].assign(Floating_PnL=pnl, Hold_Bars=hold_bars))[0][1]
                        
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
                        ep = self.models['Short_Exit_Model'].predict_proba(curr_feats[self.processor.exit_feature_cols].assign(Floating_PnL=pnl, Hold_Bars=hold_bars))[0][1]
                        
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
                'Strategy_Action': s_action, 'Strategy_Detail': s_detail,
                'User_Advice': u_action, 'User_Note': u_note,
                'K': curr_feats['K'].values[0], 'D': curr_feats['D'].values[0], 
                'MA_Slope': curr_feats['MA_Slope'].values[0], 'Time_Segment': curr_feats['Time_Segment'].values[0],
                'Settlement_Day': curr_feats['Settlement_Day'].values[0]
            })
            
        return pd.DataFrame(history_records), user_entry_idx

# ==========================================
# 4. Streamlit UI
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

# --- Layout ---
left, right = st.columns([1, 2.5])
engine = DataEngine()
models = load_models()

HIST_FILE = 'history_data.csv'

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
    
    # 資料源分頁
    tab1, tab2, tab3 = st.tabs(["🚀 即時串接", "💾 歷史管理", "📝 貼上 Excel"])
    
    df_final = pd.DataFrame()
    
    with tab1:
        st.caption("自動抓取 Anue 鉅亨網 + 讀取歷史檔")
        if st.button("🔄 立即更新資料", type="primary"):
            with st.spinner("抓取並計算中..."):
                # 1. 讀取歷史
                df_hist = pd.DataFrame()
                if os.path.exists(HIST_FILE):
                    df_hist = pd.read_csv(HIST_FILE)
                    df_hist['Time'] = pd.to_datetime(df_hist['Time'])
                
                # 2. 抓取今日
                df_real = engine.fetch_realtime_from_anue()
                
                if not df_real.empty:
                    # 3. 合併 (History + Realtime)
                    # [Critical] 確保即時資料接續在歷史資料之後，讓指標計算 (如 MA, KD) 延續
                    df_concat = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time').sort_values('Time')
                    
                    # 4. 濾除夜盤
                    df_day = engine.filter_day_session(df_concat)
                    
                    # 5. 計算指標 (這時候 KD 會基於完整的歷史資料計算，不會再是 50 了)
                    df_final = engine.calculate_indicators(df_day)
                    
                    # 6. 顯示用：只取「今天」
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    df_final = df_final[df_final['Time'].dt.strftime('%Y-%m-%d') == today_str]
                    
                    if df_final.empty:
                        st.warning("抓到了資料，但非今日日盤 (可能是假日或尚未開盤)。")
                    else:
                        st.success(f"更新成功！包含 {len(df_final)} 筆今日數據")
                else:
                    st.error("無法連線至鉅亨網，請檢查網路。")
        
        # 指標驗證區
        if not df_final.empty:
            with st.expander("🕵️‍♀️ 指標驗證區 (點擊展開)"):
                st.caption("請對照 Excel 驗證以下數值是否正確:")
                verify_df = df_final[['Time', 'Close', 'K', 'D', 'MA_Slope', 'Time_Segment', 'Settlement_Day']].copy()
                verify_df['Time'] = verify_df['Time'].dt.strftime('%H:%M')
                st.dataframe(verify_df.iloc[::-1], height=200)

    with tab2:
        st.caption("請上傳「前一日以前」的日盤資料 CSV")
        up_file = st.file_uploader("上傳歷史檔 (覆蓋)", type=['csv'])
        if up_file:
            pd.read_csv(up_file).to_csv(HIST_FILE, index=False)
            st.success("歷史檔已更新！")
            
        if st.button("💾 收盤存檔"):
            if not df_final.empty:
                save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                if os.path.exists(HIST_FILE):
                    df_old = pd.read_csv(HIST_FILE)[save_cols]
                    df_new = pd.concat([df_old, df_final[save_cols]])
                    df_new.drop_duplicates(subset='Time').to_csv(HIST_FILE, index=False)
                else:
                    df_final[save_cols].to_csv(HIST_FILE, index=False)
                st.success("已將今日資料併入歷史庫！")
            else:
                st.warning("無今日資料可存")

    with tab3:
        paste_data = st.text_area("Ctrl+V 貼上", height=150)
        if paste_data:
            st.info("請使用即時串接功能，或將貼上資料整合至 Processor")

with right:
    if models and not df_final.empty:
        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_final)
        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
        
        st.subheader("📜 歷史訊號回放")
        df_show = df_view.iloc[::-1]
        
        st.dataframe(
            df_show,
            height=400,
            column_config={
                "Time": st.column_config.DatetimeColumn("時間", format="HH:mm", width="small"),
                "Close": st.column_config.NumberColumn("收盤價", format="%d", width="small"),
                "Strategy_Action": st.column_config.TextColumn("模型策略", width="small"),
                "Strategy_Detail": st.column_config.TextColumn("策略細節", width="medium"),
                "User_Advice": st.column_config.TextColumn("持單建議", width="small"),
                "User_Note": st.column_config.TextColumn("持單細節", width="medium"),
                "K": None, "D": None, "MA_Slope": None, "Time_Segment": None, "Settlement_Day": None 
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("📊 當日走勢圖")
        df_chart = df_final.copy()
        df_chart['Time_Str'] = df_chart['Time'].dt.strftime('%H:%M')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['Time_Str'], y=df_chart['Close'], mode='lines', name='Price', line=dict(color='#1f77b4')))
        
        buys = df_view[df_view['Strategy_Action'].str.contains('買進')]
        sells = df_view[df_view['Strategy_Action'].str.contains('放空')]
        exits_long = df_view[df_view['Strategy_Action'].str.contains('❌')]
        exits_short = df_view[df_view['Strategy_Action'].str.contains('❎')]
        
        if not buys.empty: fig.add_trace(go.Scatter(x=buys['Time'].dt.strftime('%H:%M'), y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='Buy'))
        if not sells.empty: fig.add_trace(go.Scatter(x=sells['Time'].dt.strftime('%H:%M'), y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='Sell'))
        if not exits_long.empty: fig.add_trace(go.Scatter(x=exits_long['Time'].dt.strftime('%H:%M'), y=exits_long['Close'], mode='markers', marker=dict(symbol='x', size=10, color='red'), name='Exit'))
        if not exits_short.empty: fig.add_trace(go.Scatter(x=exits_short['Time'].dt.strftime('%H:%M'), y=exits_short['Close'], mode='markers', marker=dict(symbol='x', size=10, color='green'), name='Exit'))
        
        if entry_idx != -1 and entry_idx in df_chart.index:
            entry_row = df_chart.loc[entry_idx]
            color = 'red' if u_pos == "多單 (Long)" else 'green'
            fig.add_trace(go.Scatter(x=[entry_row['Time_Str']], y=[entry_row['Close']], mode='markers', marker=dict(symbol='star', size=15, color=color), name='My Entry'))

        fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=350, xaxis_type='category')
        st.plotly_chart(fig, use_container_width=True)
        
    elif models is None:
        st.error("⚠️ 模型載入失敗")
    else:
        st.info("👈 請點擊「立即更新資料」開始")
