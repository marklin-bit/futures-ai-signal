import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os
import io
from datetime import datetime, time

# 1. 網頁設定
st.set_page_config(page_title="AI 交易訊號戰情室", layout="wide", initial_sidebar_state="expanded")

# CSS 美化
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心類別定義
# ==========================================
class DataProcessor:
    def __init__(self, df):
        self.raw_df = df
        self.feature_cols = [
            'Bandwidth', 'MA_Slope', 'Bandwidth_Rate', 'Rel_Volume',
            'K', 'D', 'Position_in_Channel', 'Volatility', 
            'K_Strength', 'Body_Ratio', 'Week', 'Settlement_Day', 'Time_Segment'
        ]
        self.rename_map = {
            '布林通道寬度': 'Bandwidth', 'MA斜率\n0平/1上/-1下': 'MA_Slope', 'MA斜率': 'MA_Slope',
            '布林帶寬度變化率': 'Bandwidth_Rate', '相對成交量': 'Rel_Volume',
            'K(36,3)': 'K', 'K值': 'K', 'D(36,3)': 'D', 'D值': 'D',
            '收盤時\n通道位置': 'Position_in_Channel', '通道位置': 'Position_in_Channel',
            '波動率': 'Volatility', 'K 棒\n相對強度': 'K_Strength', 'K棒強度': 'K_Strength',
            '實體佔比': 'Body_Ratio', 'Week': 'Week', '星期': 'Week',
            '結算日\n(0/1周結算/2月結算)': 'Settlement_Day', '結算日': 'Settlement_Day',
            '時段\n(0盤初/1盤中/2盤尾)': 'Time_Segment', '時段': 'Time_Segment',
            '單別\n1多單/2空單': 'Order_Type', '動作\n0無/1買進/2持單/3賣出': 'Action',
            '收盤價': 'Close', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low',
            '收盤時間': 'Time', '時間': 'Time'
        }
        self.exit_feature_cols = self.feature_cols + ['Floating_PnL', 'Hold_Bars']

    def validate_time_continuity(self, df):
        if 'Time' not in df.columns: return [], "No Time Column"
        try:
            time_series = pd.to_datetime(df['Time'])
            diffs = time_series.diff()
            discontinuous_indices = []
            for i in range(1, len(diffs)):
                delta = diffs.iloc[i]
                if delta.total_seconds() != 300:
                    curr = time_series.iloc[i].strftime('%H:%M')
                    prev = time_series.iloc[i-1].strftime('%H:%M')
                    discontinuous_indices.append(f"{prev} -> {curr}")
            return discontinuous_indices, None
        except: return [], "Error"

    def process(self):
        if self.raw_df is None or self.raw_df.empty: return pd.DataFrame(), [], []
        df = self.raw_df.copy()
        df.columns = df.columns.astype(str)
        df.rename(columns=lambda x: x.replace('\n', '').strip(), inplace=True)
        
        clean_map = {}
        for col in df.columns:
            if col in self.rename_map: clean_map[col] = self.rename_map[col]
            else:
                for k, v in self.rename_map.items():
                    if k in col: clean_map[col] = v; break
        df.rename(columns=clean_map, inplace=True)
        
        missing = []
        for col in self.feature_cols:
            if col not in df.columns: missing.append(col); df[col] = 0
            else: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        disc = []
        if 'Time' in df.columns: disc, _ = self.validate_time_continuity(df)
        
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, missing, disc

class StrategyEngine:
    def __init__(self, df, models, params):
        self.df = df
        self.models = models
        self.params = params
        self.processor = DataProcessor(None) # Helper

    def find_entry_info(self, entry_time_obj):
        """
        根據時間物件 (datetime.time) 尋找對應的 Index 和 Close Price
        """
        if entry_time_obj is None: return -1, 0.0
        time_str = entry_time_obj.strftime("%H:%M")
        mask = self.df['Time'].astype(str).str.contains(time_str, na=False)
        matches = self.df[mask]
        if not matches.empty:
            idx = matches.index[-1]
            price = matches.loc[idx, 'Close']
            return idx, price
        return -1, 0.0

    def run_historical_review(self, user_pos_type, entry_time_obj):
        # 策略模擬變數
        strat_pos = 0 
        strat_entry_price = 0.0
        strat_entry_index = 0
        
        # 使用者設定
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        
        # 自動查找成本與 Index
        user_entry_idx = -1
        user_cost = 0.0
        if u_pos != "Empty":
            user_entry_idx, user_cost = self.find_entry_info(entry_time_obj)

        history_records = []
        X_all = self.df[self.processor.feature_cols]
        
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_features = X_all.iloc[[i]]
            current_close = current_bar.get('Close', 0)
            current_time = current_bar.get('Time', f"K_{i}")
            
            # 預先計算進場機率
            prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
            prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
            
            trend_str = f"(多:{prob_long:.0%}/空:{prob_short:.0%})"

            # --- 1. 計算策略模擬 (Strategy Simulation) [🔴/🟢 圓形系統] ---
            strat_action = "⚪ 觀望"
            strat_detail = ""

            if strat_pos == 0:
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    strat_pos = 1
                    strat_entry_price = current_close
                    strat_entry_index = i
                    strat_action = "🔴 買進"
                    strat_detail = f"多 {prob_long:.0%} {trend_str}"
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    strat_pos = -1
                    strat_entry_price = current_close
                    strat_entry_index = i
                    strat_action = "🟢 放空"
                    strat_detail = f"空 {prob_short:.0%} {trend_str}"
                else:
                    strat_detail = f"{trend_str}"

            elif strat_pos == 1:
                pnl = current_close - strat_entry_price
                if pnl <= -self.params['hard_stop']:
                    strat_pos = 0
                    strat_action = "💥 停損"
                    strat_detail = f"損 {pnl:.0f} {trend_str}"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = pnl
                    exit_feats['Hold_Bars'] = i - strat_entry_index
                    exit_feats = exit_feats[self.processor.exit_feature_cols]
                    prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if prob > self.params['exit_threshold']:
                        strat_pos = 0
                        strat_action = "🟢 多出" # 多單出場(賣) -> 綠色
                        strat_detail = f"出場率 {prob:.0%} {trend_str}"
                    else:
                        strat_action = "⏳ 續抱"
                        strat_detail = f"帳面 {pnl:.0f} {trend_str}"

            elif strat_pos == -1:
                pnl = strat_entry_price - current_close
                if pnl <= -self.params['hard_stop']:
                    strat_pos = 0
                    strat_action = "💥 停損"
                    strat_detail = f"損 {pnl:.0f} {trend_str}"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = pnl
                    exit_feats['Hold_Bars'] = i - strat_entry_index
                    exit_feats = exit_feats[self.processor.exit_feature_cols]
                    prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if prob > self.params['exit_threshold']:
                        strat_pos = 0
                        strat_action = "🔴 空出" # 空單出場(買) -> 紅色
                        strat_detail = f"出場率 {prob:.0%} {trend_str}"
                    else:
                        strat_action = "⏳ 續抱"
                        strat_detail = f"帳面 {pnl:.0f} {trend_str}"

            # --- 2. 計算使用者持單建議 (User Advice) [🟥/🟩 方形系統] ---
            user_advice = "-"
            user_note = ""
            
            if u_pos == "Empty":
                user_advice = "未持單"
                user_note = "-"
            
            elif user_entry_idx == -1:
                user_advice = "時間未對上"
                user_note = "查無此K棒"

            elif i < user_entry_idx:
                user_advice = "未持單"
                user_note = "-"
            
            elif i == user_entry_idx:
                if u_pos == "Long":
                    user_advice = "🟥 多單進場" 
                else:
                    user_advice = "🟩 空單進場"
                user_note = f"成本 {user_cost:.0f}"

            else:
                # 持倉中
                current_bars_held = i - user_entry_idx
                
                if u_pos == "Long":
                    u_pnl = current_close - user_cost
                    if u_pnl <= -self.params['hard_stop']:
                        user_advice = "💥 停損"
                        user_note = f"{u_pnl:.0f}"
                    else:
                        u_exit_feats = current_features.copy()
                        u_exit_feats['Floating_PnL'] = u_pnl
                        u_exit_feats['Hold_Bars'] = current_bars_held
                        u_exit_feats = u_exit_feats[self.processor.exit_feature_cols]
                        u_prob = self.models['Long_Exit_Model'].predict_proba(u_exit_feats)[0][1]
                        hold_conf = 1.0 - u_prob
                        
                        # 格式: 帳面XX(續:X%/多:X%/空:X%)
                        status_str = f"帳面{u_pnl:.0f}(續:{hold_conf:.0%}/多:{prob_long:.0%}/空:{prob_short:.0%})"
                        
                        if u_prob > self.params['exit_threshold']:
                            user_advice = "🏃 出場"
                            user_note = f"出場率 {u_prob:.0%} {trend_str}"
                        else:
                            if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                                user_advice = "🟥 加碼"
                                user_note = status_str
                            else:
                                user_advice = "🟥 續抱"
                                user_note = status_str

                elif u_pos == "Short":
                    u_pnl = user_cost - current_close
                    if u_pnl <= -self.params['hard_stop']:
                        user_advice = "💥 停損"
                        user_note = f"{u_pnl:.0f}"
                    else:
                        u_exit_feats = current_features.copy()
                        u_exit_feats['Floating_PnL'] = u_pnl
                        u_exit_feats['Hold_Bars'] = current_bars_held
                        u_exit_feats = u_exit_feats[self.processor.exit_feature_cols]
                        u_prob = self.models['Short_Exit_Model'].predict_proba(u_exit_feats)[0][1]
                        hold_conf = 1.0 - u_prob
                        
                        status_str = f"帳面{u_pnl:.0f}(續:{hold_conf:.0%}/多:{prob_long:.0%}/空:{prob_short:.0%})"
                        
                        if u_prob > self.params['exit_threshold']:
                            user_advice = "🏃 出場"
                            user_note = f"出場率 {u_prob:.0%} {trend_str}"
                        else:
                            if prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                                user_advice = "🟩 加碼"
                                user_note = status_str
                            else:
                                user_advice = "🟩 續抱"
                                user_note = status_str

            record = {
                'Time': current_time,
                'Close': current_close,
                'Strategy_Action': strat_action,
                'Strategy_Detail': strat_detail,
                'User_Advice': user_advice,
                'User_Note': user_note
            }
            history_records.append(record)
            
        return pd.DataFrame(history_records)

    def run_realtime_advice(self, user_position, entry_time_obj):
        last_idx = len(self.df) - 1
        current_features = self.df.iloc[[last_idx]][DataProcessor(None).feature_cols].copy()
        current_close = self.df.iloc[last_idx].get('Close', 0)
        
        advice = {"Action": "Wait", "Confidence": 0.0, "PnL": 0.0, "Message": "資料不足", "Type": "Neutral", "Label": "進場信心"}

        prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
        prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]

        if user_position == 'Empty':
            advice["Label"] = "進場信心"
            if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                advice.update({"Action": "Buy", "Confidence": prob_long, "Message": "🔥 多方強勢，建議買進", "Type": "Buy"})
            elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                advice.update({"Action": "Sell", "Confidence": prob_short, "Message": "⚡ 空方強勢，建議放空", "Type": "Sell"})
            else:
                advice.update({"Action": "Wait", "Confidence": max(prob_long, prob_short), "Message": f"觀望 (多:{prob_long:.2f}/空:{prob_short:.2f})", "Type": "Wait"})
        else:
            user_entry_idx, entry_price = self.find_entry_info(entry_time_obj)
            bars_held = 0
            if user_entry_idx != -1 and last_idx >= user_entry_idx:
                bars_held = last_idx - user_entry_idx
            if bars_held < 0: bars_held = 0

            if user_position == 'Long':
                floating_pnl = current_close - entry_price
                advice['PnL'] = floating_pnl
                
                if floating_pnl <= -self.params['hard_stop']:
                    advice.update({"Action": "StopLoss", "Confidence": 1.0, "Message": f"🛑 觸發硬停損 (-{self.params['hard_stop']})", "Type": "Stop", "Label": "停損觸發"})
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = bars_held
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                    
                    exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    
                    if exit_prob > self.params['exit_threshold']:
                        advice.update({"Action": "Exit", "Confidence": exit_prob, "Message": f"🚀 建議多單出場 (機率 {exit_prob:.0%})", "Type": "Exit", "Label": "出場機率"})
                    else:
                        hold_conf = 1.0 - exit_prob
                        if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                            advice.update({"Action": "Hold+", "Confidence": prob_long, "Message": "⚓ 續抱且出現多方訊號 (🔥可加碼)", "Type": "Buy", "Label": "加碼信心"})
                        else:
                            advice.update({"Action": "Hold", "Confidence": hold_conf, "Message": f"⚓ 建議續抱 (安心度 {hold_conf:.0%})", "Type": "Hold", "Label": "續抱信心"})

            elif user_position == 'Short':
                floating_pnl = entry_price - current_close
                advice['PnL'] = floating_pnl
                
                if floating_pnl <= -self.params['hard_stop']:
                    advice.update({"Action": "StopLoss", "Confidence": 1.0, "Message": f"🛑 觸發硬停損 (-{self.params['hard_stop']})", "Type": "Stop", "Label": "停損觸發"})
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = bars_held
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                    
                    exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    
                    if exit_prob > self.params['exit_threshold']:
                        advice.update({"Action": "Exit", "Confidence": exit_prob, "Message": f"🚀 建議空單出場 (機率 {exit_prob:.0%})", "Type": "Exit", "Label": "出場機率"})
                    else:
                        hold_conf = 1.0 - exit_prob
                        if prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                            advice.update({"Action": "Hold+", "Confidence": prob_short, "Message": "⚓ 續抱且出現空方訊號 (🔥可加碼)", "Type": "Sell", "Label": "加碼信心"})
                        else:
                            advice.update({"Action": "Hold", "Confidence": hold_conf, "Message": f"⚓ 建議續抱 (安心度 {hold_conf:.0%})", "Type": "Hold", "Label": "續抱信心"})

        return advice

# ==========================================
# 3. 載入模型
# ==========================================
@st.cache_resource
def load_models():
    names = ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']
    loaded = {}
    paths = ['', 'models/']
    for name in names:
        m = None
        for p in paths:
            if os.path.exists(f"{p}{name}.pkl"):
                try: m = joblib.load(f"{p}{name}.pkl"); break
                except: pass
        if m: loaded[name] = m
        else: return None
    return loaded

# ==========================================
# 4. 網頁介面主邏輯
# ==========================================
left_col, right_col = st.columns([1, 2.5])
models = load_models()

# --- 左側：輸入與控制 ---
with left_col:
    st.subheader("🛠️ 數據與參數")
    with st.expander("⚙️ 參數設定", expanded=False):
        entry_threshold = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        exit_threshold = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        hard_stop = st.number_input("硬停損點數", value=100, step=10)

    st.markdown("##### 👤 目前真實部位")
    st.caption("設定後，右側表格將顯示針對此部位的歷史建議")
    user_pos_type = st.radio("持倉狀態", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"], label_visibility="collapsed")
    
    user_entry_time = None
    if user_pos_type != "空手 (Empty)":
        user_entry_time = st.time_input("買進時間 (每5分一跳)", value=time(9, 0), step=300, help="系統會自動抓取該時間的收盤價作為成本")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📝 貼上資料", "🔄 即時串接"])
    df_input = None
    with tab1:
        st.caption("請從 Excel 複製含標題數據")
        paste_data = st.text_area("Ctrl+V 貼上區", height=250, label_visibility="collapsed")
        if paste_data:
            try: df_input = pd.read_csv(io.StringIO(paste_data), sep='\t')
            except: st.error("格式錯誤")
    with tab2: st.info("🚧 開發中")

# --- 右側：歷史訊號列表 (優先) ---
with right_col:
    if models is None:
        st.error("⚠️ 模型載入失敗")
    elif df_input is not None and not df_input.empty:
        processor = DataProcessor(df_input)
        df_clean, missing_cols, discontinuities = processor.process()

        if missing_cols:
            st.error(f"❌ 缺少欄位：{missing_cols}")
        else:
            if discontinuities:
                with st.expander(f"⚠️ 時間不連續警示 ({len(discontinuities)})"):
                    st.write(discontinuities[:5])

            params = {'entry_threshold': entry_threshold, 'exit_threshold': exit_threshold, 'hard_stop': hard_stop}
            engine = StrategyEngine(df_clean, models, params)
            
            df_history = engine.run_historical_review(user_pos_type, user_entry_time)
            
            # 取得即時建議
            pos_map_key = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}[user_pos_type]
            advice = engine.run_realtime_advice(pos_map_key, user_entry_time)

            # --- Dashboard ---
            st.markdown("---")
            last_row = df_clean.iloc[-1]
            
            m1, m2, m3 = st.columns([1, 1.5, 1.5])
            m1.metric("📊 最新時間", str(last_row.get('Time', 'N/A'))[-5:]) 
            
            delta_color = "off"
            if advice['Type'] in ['Buy', 'Exit']: delta_color = "normal"
            elif advice['Type'] in ['Sell', 'Stop']: delta_color = "inverse"
            m2.metric("🤖 AI 決策", advice['Type'], delta=advice['Message'], delta_color=delta_color)
            
            pnl_show = f"{advice['PnL']:.0f}" if user_pos_type != "空手 (Empty)" else "-"
            m3.metric(f"🎯 {advice['Label']}/損益", f"{advice['Confidence']:.0%}", delta=pnl_show)

            # --- A. 歷史訊號列表 (置頂) ---
            st.subheader("📜 歷史訊號回放")
            
            c_sort, _ = st.columns([1, 2])
            sort_order = c_sort.radio("排序方式", ["時間：新 → 舊 (倒序)", "時間：舊 → 新 (正序)"], horizontal=True, label_visibility="collapsed")
            
            df_show = df_history.copy()
            if "新 → 舊" in sort_order:
                df_show = df_show.iloc[::-1] # 倒序
            
            st.dataframe(
                df_show,
                use_container_width=True,
                height=400,
                column_config={
                    "Time": st.column_config.TextColumn("時間", width="small"),
                    "Close": st.column_config.NumberColumn("收盤價", format="%.0f", width="small"),
                    "Strategy_Action": st.column_config.TextColumn("模型策略", help="若 AI 全自動交易的操作", width="small"),
                    "Strategy_Detail": st.column_config.TextColumn("策略細節", width="medium"),
                    "User_Advice": st.column_config.TextColumn("持單建議", help="針對左側設定的部位給出的建議", width="small"),
                    "User_Note": st.column_config.TextColumn("持單細節", width="medium")
                },
                hide_index=True
            )

            # --- B. 視覺化圖表 ---
            st.markdown("---")
            st.subheader("📊 近 60 根 K 棒走勢")
            
            df_chart = df_clean.tail(60)
            df_hist_chart = df_history.tail(60)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart['Time'], y=df_chart['Close'], mode='lines+markers', name='Price', line=dict(color='#1f77b4', width=2)))
            
            # 策略點標記
            buys = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('買進')]
            sells = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('放空')]
            exits_long = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('多出')]
            exits_short = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('空出')]
            
            # 紅買/綠賣
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys['Time'], y=buys['Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=15, color='red')))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells['Time'], y=sells['Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=15, color='green')))
            if not exits_long.empty:
                fig.add_trace(go.Scatter(x=exits_long['Time'], y=exits_long['Close'], mode='markers', name='Exit Long', marker=dict(symbol='x', size=12, color='green')))
            if not exits_short.empty:
                fig.add_trace(go.Scatter(x=exits_short['Time'], y=exits_short['Close'], mode='markers', name='Exit Short', marker=dict(symbol='x', size=12, color='red')))
            
            # [Added] 標記真實部位進場點
            real_entry_idx, _ = engine.find_entry_info(user_entry_time)
            
            if real_entry_idx != -1 and real_entry_idx in df_chart.index:
                entry_row = df_clean.loc[real_entry_idx]
                
                # 設定標記樣式 (紅漲綠跌)
                marker_symbol = 'star'
                marker_color = 'red' if user_pos_type == "多單 (Long)" else 'green'
                marker_name = 'My Entry'
                
                if user_pos_type != "空手 (Empty)":
                    fig.add_trace(go.Scatter(
                        x=[entry_row['Time']], 
                        y=[entry_row['Close']], 
                        mode='markers', 
                        name=marker_name, 
                        marker=dict(symbol=marker_symbol, size=20, color=marker_color, line=dict(width=2, color='white'))
                    ))

            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis_type='category')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 請在左側貼上資料以開始")
