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
        
        # 抓取最近 300 筆 (足夠涵蓋今日日盤 + 昨日夜盤)
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

    def calculate_indicators(self, df, mode='day'):
        """
        依照使用者指定的公式計算 13 個特徵
        mode: 'day' (日盤模式) 或 'full' (全盤模式)
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
        
        # 5 & 6. KD (36, 3)
        rsv_window = 36
        l_min = L.rolling(rsv_window).min()
        h_max = H.rolling(rsv_window).max()
        rsv = (C - l_min) / (h_max - l_min) # 0.0 ~ 1.0
        
        k_vals = [0.5] * len(df)
        d_vals = [0.5] * len(df)
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
        
        # 8. 波動率
        df['Volatility'] = (H - L) / C * 100
        
        # 9. K棒強度
        df['K_Strength'] = (C - O) / O * 100
        
        # 10. 實體佔比
        hl_range = (H - L).replace(0, 1)
        df['Body_Ratio'] = (C - O).abs() / hl_range
        
        # 11. 星期
        df['Week'] = df['Time'].dt.weekday + 1
        
        if mode == 'full':
            # 全盤模式：強制設定
            df['Settlement_Day'] = 0
            df['Time_Segment'] = 1
        else:
            # 日盤模式：正常計算
            # 12. 結算日
            def get_settlement(row):
                score = 0
                d = row['Time'].date()
                if d.weekday() in [2, 4]: score += 1
                if str(d) in SETTLEMENT_DATES_2026: score += 1
                return score
            df['Settlement_Day'] = df.apply(get_settlement, axis=1)
            
            # 13. 時段
            def get_segment(t):
                hm = t.hour * 100 + t.minute
                if hm <= 930: return 0
                elif hm <= 1200: return 1
                else: return 2
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
        
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        user_entry_idx, user_cost = self.find_entry_info(entry_time_obj) if u_pos != "Empty" else (-1, 0.0)
        
        s_pos, s_price, s_idx = 0, 0.0, 0
        
        for i in range(len(self.df)):
            curr_time = self.df.iloc[i]['Time']
            curr_close = self.df.iloc[i]['Close']
            curr_feats = X_all.iloc[[i]]
            
            p_long = self.models['Long_Entry_Model'].predict_proba(curr_feats)[0][1]
            p_short = self.models['Short_Entry_Model'].predict_proba(curr_feats)[0][1]
            
            trend = f"(多:{p_long:.0%}/空:{p_short:.0%})"
            
            # 1. 策略模擬
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
                    # [Fix] 先 assign 欄位，再選取 exit_feature_cols
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
                    # [Fix] 先 assign 欄位，再選取 exit_feature_cols
                    curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=i-s_idx)
                    exit_prob = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                    
                    if exit_prob > self.params['exit']:
                        s_pos, s_action, s_detail = 0, "❎ 空出", f"帳{pnl:.0f}(出:{exit_prob:.0%})"
                    else:
                        s_action, s_detail = "⏳ 續抱", f"帳{pnl:.0f}(續:{1-exit_prob:.0%})"

            # 2. 持單建議
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
                        # [Fix] 先 assign 欄位，再選取 exit_feature_cols
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Long_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        
                        detail = f"帳面{pnl:.0f}(出:{ep:.0%}{trend})"
                        if ep > self.params['exit']:
                            u_action, u_note = "❌ 出場", detail
                        elif p_long > self.params['entry'] and p_long > p_short:
                            u_action, u_note = "🔥 加碼", detail
                        else:
                            u_action, u_note = "⏳ 續抱", detail
                elif u_pos == "Short":
                    pnl = user_cost - curr_close
                    if pnl <= -self.params['stop']:
                        u_action, u_note = "💥 停損", f"{pnl:.0f}"
                    else:
                        # [Fix] 先 assign 欄位，再選取 exit_feature_cols
                        curr_feats_exit = curr_feats.assign(Floating_PnL=pnl, Hold_Bars=hold_bars)
                        ep = self.models['Short_Exit_Model'].predict_proba(curr_feats_exit[self.processor.exit_feature_cols])[0][1]
                        
                        detail = f"帳面{pnl:.0f}(出:{ep:.0%}{trend})"
                        if ep > self.params['exit']:
                            u_action, u_note = "❎ 出場", detail
                        elif p_short > self.params['entry'] and p_short > p_long:
                            u_action, u_note = "🔥 加碼", detail
                        else:
                            u_action, u_note = "⏳ 續抱", detail

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

# 檔案路徑設定
HIST_FILE_DAY = 'history_data_day.csv'
HIST_FILE_FULL = 'history_data_full.csv'

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
    
    # 資料源分頁 (擴增為 5 個)
    tab_r_day, tab_h_day, tab_r_full, tab_h_full, tab_paste = st.tabs(["🌞 即時(日)", "💾 歷史(日)", "🌙 即時(全)", "💾 歷史(全)", "📝 貼上"])
    
    df_final = pd.DataFrame()
    current_mode = 'day' # 用來標記當前是哪個分頁觸發的計算，影響顯示標題
    
    # 1. 即時串接 (日)
    with tab_r_day:
        st.caption("日盤模式：自動濾除夜盤，指標延續昨日收盤。")
        if st.button("🔄 更新日盤資料", type="primary", key="btn_real_day"):
            current_mode = 'day'
            with st.spinner("抓取並計算中..."):
                df_hist = pd.read_csv(HIST_FILE_DAY) if os.path.exists(HIST_FILE_DAY) else pd.DataFrame()
                if not df_hist.empty: df_hist['Time'] = pd.to_datetime(df_hist['Time'])
                
                df_real = engine.fetch_realtime_from_anue()
                
                if not df_real.empty:
                    df_concat = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time').sort_values('Time')
                    df_day = engine.filter_day_session(df_concat)
                    df_final = engine.calculate_indicators(df_day, mode='day')
                    
                    # 只留今天
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    df_final = df_final[df_final['Time'].dt.strftime('%Y-%m-%d') == today_str]
                    
                    if df_final.empty: st.warning("抓到資料但非今日日盤。")
                    else: st.success(f"日盤更新成功！({len(df_final)} 筆)")
                else: st.error("連線失敗")

    # 2. 歷史管理 (日)
    with tab_h_day:
        st.caption("上傳「純日盤」歷史檔")
        up_day = st.file_uploader("上傳 history_data_day.csv", type=['csv'], key="up_day")
        if up_day:
            pd.read_csv(up_day).to_csv(HIST_FILE_DAY, index=False)
            st.success("日盤歷史檔已更新")
        if st.button("💾 存檔 (併入今日日盤)", key="save_day"):
            if not df_final.empty and current_mode == 'day':
                save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                if os.path.exists(HIST_FILE_DAY):
                    df_old = pd.read_csv(HIST_FILE_DAY)[save_cols]
                    df_new = pd.concat([df_old, df_final[save_cols]])
                    df_new.drop_duplicates(subset='Time').to_csv(HIST_FILE_DAY, index=False)
                else:
                    df_final[save_cols].to_csv(HIST_FILE_DAY, index=False)
                st.success("存檔成功")
            else:
                st.warning("無資料可存 (請先執行即時更新)")

    # 3. 即時串接 (全)
    with tab_r_full:
        st.caption("全盤模式：包含夜盤，參考用 (時段=1, 結算=0)。")
        if st.button("🔄 更新全盤資料", key="btn_real_full"):
            current_mode = 'full'
            with st.spinner("抓取中..."):
                df_hist = pd.read_csv(HIST_FILE_FULL) if os.path.exists(HIST_FILE_FULL) else pd.DataFrame()
                if not df_hist.empty: df_hist['Time'] = pd.to_datetime(df_hist['Time'])
                
                df_real = engine.fetch_realtime_from_anue()
                
                if not df_real.empty:
                    # 全盤模式不濾除夜盤，直接拼接
                    df_concat = pd.concat([df_hist, df_real]).drop_duplicates(subset='Time').sort_values('Time')
                    
                    # 計算指標 (mode='full')
                    df_final = engine.calculate_indicators(df_concat, mode='full')
                    
                    # 顯示最近 100 筆 (因為全盤跨日長，顯示太多會亂)
                    df_final = df_final.tail(100)
                    
                    if df_final.empty: st.warning("無資料")
                    else: st.success(f"全盤更新成功！({len(df_final)} 筆)")
                else: st.error("連線失敗")

    # 4. 歷史管理 (全)
    with tab_h_full:
        st.caption("上傳「全盤」歷史檔")
        up_full = st.file_uploader("上傳 history_data_full.csv", type=['csv'], key="up_full")
        if up_full:
            pd.read_csv(up_full).to_csv(HIST_FILE_FULL, index=False)
            st.success("全盤歷史檔已更新")
        if st.button("💾 存檔 (併入今日全盤)", key="save_full"):
            if not df_final.empty and current_mode == 'full':
                save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                if os.path.exists(HIST_FILE_FULL):
                    df_old = pd.read_csv(HIST_FILE_FULL)[save_cols]
                    df_new = pd.concat([df_old, df_final[save_cols]])
                    df_new.drop_duplicates(subset='Time').to_csv(HIST_FILE_FULL, index=False)
                else:
                    df_final[save_cols].to_csv(HIST_FILE_FULL, index=False)
                st.success("全盤存檔成功")
            else:
                st.warning("無資料可存")

    # 5. 貼上 (保留)
    with tab_paste:
        paste_data = st.text_area("Ctrl+V 貼上", height=150)
        if paste_data: st.info("建議使用自動串接")

with right:
    if models and not df_final.empty:
        strat = StrategyEngine(models, {'entry': p_entry, 'exit': p_exit, 'stop': p_stop}, df_final)
        df_view, entry_idx = strat.run_analysis(u_pos, u_time)
        
        mode_title = "🌞 日盤" if current_mode == 'day' else "🌙 全盤"
        st.subheader(f"📜 {mode_title}訊號回放")
        
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
        
        st.subheader(f"📊 {mode_title}走勢圖")
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
