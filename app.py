import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os
import io

# 1. 網頁設定
st.set_page_config(page_title="AI 交易訊號戰情室", layout="wide", initial_sidebar_state="expanded")

# CSS 美化
st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
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

    def run_historical_review(self, user_pos_type, user_cost, user_bars):
        """
        同時計算：
        1. 策略模擬 (Auto Strategy): AI 自己玩會怎麼做 (空手開始)
        2. 持單建議 (User Advice): 假設使用者在該時間點持有特定部位，AI 建議為何 (含加碼偵測)
        """
        # 策略模擬變數
        strat_pos = 0 
        strat_entry_price = 0.0
        strat_entry_index = 0
        
        # 使用者設定映射
        pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
        u_pos = pos_map.get(user_pos_type, "Empty")
        
        history_records = []
        X_all = self.df[self.processor.feature_cols]
        
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_features = X_all.iloc[[i]]
            current_close = current_bar.get('Close', 0)
            current_time = current_bar.get('Time', f"K_{i}")
            
            # 預先計算進場機率 (每一根都算，為了加碼判斷)
            prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
            prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]

            # --- 1. 計算策略模擬 (Strategy Simulation) ---
            strat_action = "觀望"
            strat_detail = ""

            if strat_pos == 0:
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    strat_pos = 1
                    strat_entry_price = current_close
                    strat_entry_index = i
                    strat_action = "🔥 買進"
                    strat_detail = f"多 {prob_long:.0%}"
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    strat_pos = -1
                    strat_entry_price = current_close
                    strat_entry_index = i
                    strat_action = "⚡ 放空"
                    strat_detail = f"空 {prob_short:.0%}"
                else:
                    strat_detail = f"多:{prob_long:.0%} / 空:{prob_short:.0%}"

            elif strat_pos == 1:
                pnl = current_close - strat_entry_price
                if pnl <= -self.params['hard_stop']:
                    strat_pos = 0
                    strat_action = "🛑 停損"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = pnl
                    exit_feats['Hold_Bars'] = i - strat_entry_index
                    exit_feats = exit_feats[self.processor.exit_feature_cols]
                    prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if prob > self.params['exit_threshold']:
                        strat_pos = 0
                        strat_action = "🟢 出場"
                        # 修正: 顯示出場率並保留多空信心
                        strat_detail = f"出場 {prob:.0%} (多:{prob_long:.0%}/空:{prob_short:.0%})"
                    else:
                        strat_action = "續抱"
                        strat_detail = f"帳 {pnl:.0f}"

            elif strat_pos == -1:
                pnl = strat_entry_price - current_close
                if pnl <= -self.params['hard_stop']:
                    strat_pos = 0
                    strat_action = "🛑 停損"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = pnl
                    exit_feats['Hold_Bars'] = i - strat_entry_index
                    exit_feats = exit_feats[self.processor.exit_feature_cols]
                    prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if prob > self.params['exit_threshold']:
                        strat_pos = 0
                        strat_action = "🔴 出場"
                        # 修正: 顯示出場率並保留多空信心
                        strat_detail = f"出場 {prob:.0%} (多:{prob_long:.0%}/空:{prob_short:.0%})"
                    else:
                        strat_action = "續抱"
                        strat_detail = f"帳 {pnl:.0f}"

            # --- 2. 計算使用者持單建議 (含加碼偵測) ---
            user_advice = "-"
            user_note = ""
            
            if u_pos == "Empty":
                # 修正: 空手時不顯示進場訊號，因為這是"持單建議"欄位
                user_advice = "無持單"
                user_note = "-"
            
            elif u_pos == "Long":
                u_pnl = current_close - user_cost
                if u_pnl <= -self.params['hard_stop']:
                    user_advice = "🛑 觸發硬停損"
                    user_note = f"{u_pnl:.0f}"
                else:
                    u_exit_feats = current_features.copy()
                    u_exit_feats['Floating_PnL'] = u_pnl
                    u_exit_feats['Hold_Bars'] = user_bars # 壓力測試值
                    u_exit_feats = u_exit_feats[self.processor.exit_feature_cols]
                    u_prob = self.models['Long_Exit_Model'].predict_proba(u_exit_feats)[0][1]
                    
                    if u_prob > self.params['exit_threshold']:
                        user_advice = "🚀 建議出場"
                        user_note = f"機率 {u_prob:.0%}"
                    else:
                        # 續抱狀態，檢查是否可加碼
                        if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                            user_advice = "⚓ 續抱 (🔥可加碼)"
                            user_note = f"信心 {prob_long:.0%}"
                        else:
                            user_advice = "⚓ 建議續抱"
                            user_note = f"帳 {u_pnl:.0f}"

            elif u_pos == "Short":
                u_pnl = user_cost - current_close
                if u_pnl <= -self.params['hard_stop']:
                    user_advice = "🛑 觸發硬停損"
                    user_note = f"{u_pnl:.0f}"
                else:
                    u_exit_feats = current_features.copy()
                    u_exit_feats['Floating_PnL'] = u_pnl
                    u_exit_feats['Hold_Bars'] = user_bars
                    u_exit_feats = u_exit_feats[self.processor.exit_feature_cols]
                    u_prob = self.models['Short_Exit_Model'].predict_proba(u_exit_feats)[0][1]
                    
                    if u_prob > self.params['exit_threshold']:
                        user_advice = "🚀 建議出場"
                        user_note = f"機率 {u_prob:.0%}"
                    else:
                        # 續抱狀態，檢查是否可加碼
                        if prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                            user_advice = "⚓ 續抱 (🔥可加碼)"
                            user_note = f"信心 {prob_short:.0%}"
                        else:
                            user_advice = "⚓ 建議續抱"
                            user_note = f"帳 {u_pnl:.0f}"

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

    def run_realtime_advice(self, user_position, entry_price, bars_held):
        last_idx = len(self.df) - 1
        current_features = self.df.iloc[[last_idx]][DataProcessor(None).feature_cols].copy()
        current_close = self.df.iloc[last_idx].get('Close', 0)
        
        advice = {"Action": "Wait", "Confidence": 0.0, "PnL": 0.0, "Message": "資料不足", "Type": "Neutral"}

        # 預先計算進場信心 (供加碼判斷用)
        prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
        prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]

        if user_position == 'Empty':
            if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                advice.update({"Action": "Buy", "Confidence": prob_long, "Message": "🔥 多方訊號強勢，建議買進", "Type": "Buy"})
            elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                advice.update({"Action": "Sell", "Confidence": prob_short, "Message": "⚡ 空方訊號強勢，建議放空", "Type": "Sell"})
            else:
                advice.update({"Action": "Wait", "Confidence": max(prob_long, prob_short), "Message": f"觀望 (多:{prob_long:.2f}/空:{prob_short:.2f})", "Type": "Wait"})

        elif user_position == 'Long':
            floating_pnl = current_close - entry_price
            advice['PnL'] = floating_pnl
            
            if floating_pnl <= -self.params['hard_stop']:
                advice.update({"Action": "StopLoss", "Confidence": 1.0, "Message": f"🛑 觸發硬停損 (-{self.params['hard_stop']})", "Type": "Stop"})
            else:
                exit_feats = current_features.copy()
                exit_feats['Floating_PnL'] = floating_pnl
                exit_feats['Hold_Bars'] = bars_held
                exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                
                exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                
                if exit_prob > self.params['exit_threshold']:
                    advice.update({"Action": "Exit", "Confidence": exit_prob, "Message": f"🚀 AI 建議多單出場 (機率 {exit_prob:.0%})", "Type": "Exit"})
                else:
                    # 檢查加碼
                    if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                        advice.update({"Action": "Hold+", "Confidence": prob_long, "Message": "⚓ 續抱且出現多方訊號 (🔥可加碼)", "Type": "Buy"})
                    else:
                        advice.update({"Action": "Hold", "Confidence": 1-exit_prob, "Message": f"⚓ AI 建議續抱 (出場率 {exit_prob:.0%})", "Type": "Hold"})

        elif user_position == 'Short':
            floating_pnl = entry_price - current_close
            advice['PnL'] = floating_pnl
            
            if floating_pnl <= -self.params['hard_stop']:
                advice.update({"Action": "StopLoss", "Confidence": 1.0, "Message": f"🛑 觸發硬停損 (-{self.params['hard_stop']})", "Type": "Stop"})
            else:
                exit_feats = current_features.copy()
                exit_feats['Floating_PnL'] = floating_pnl
                exit_feats['Hold_Bars'] = bars_held
                exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                
                exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                
                if exit_prob > self.params['exit_threshold']:
                    advice.update({"Action": "Exit", "Confidence": exit_prob, "Message": f"🚀 AI 建議空單出場 (機率 {exit_prob:.0%})", "Type": "Exit"})
                else:
                    # 檢查加碼
                    if prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                        advice.update({"Action": "Hold+", "Confidence": prob_short, "Message": "⚓ 續抱且出現空方訊號 (🔥可加碼)", "Type": "Sell"})
                    else:
                        advice.update({"Action": "Hold", "Confidence": 1-exit_prob, "Message": f"⚓ AI 建議續抱 (出場率 {exit_prob:.0%})", "Type": "Hold"})

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
    user_entry_price = 0.0
    user_bars_held = 0
    if user_pos_type != "空手 (Empty)":
        c1, c2 = st.columns(2)
        user_entry_price = c1.number_input("成本", value=17500.0, step=1.0)
        user_bars_held = c2.number_input("K棒數", value=1, step=1, min_value=1)

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
            
            # 執行回測與建議計算
            df_history = engine.run_historical_review(user_pos_type, user_entry_price, user_bars_held)

            # --- A. 歷史訊號列表 (置頂) ---
            st.subheader("📜 歷史訊號回放")
            
            # 排序控制
            c_sort, _ = st.columns([1, 2])
            sort_order = c_sort.radio("排序方式", ["時間：新 → 舊 (倒序)", "時間：舊 → 新 (正序)"], horizontal=True, label_visibility="collapsed")
            
            df_show = df_history.copy()
            if "新 → 舊" in sort_order:
                df_show = df_show.iloc[::-1] # 倒序
            
            # 使用 column_config 優化顯示
            st.dataframe(
                df_show,
                use_container_width=True,
                height=400,
                column_config={
                    "Time": "時間",
                    "Close": "收盤價",
                    "Strategy_Action": st.column_config.TextColumn("AI 自動策略", help="若 AI 全自動交易的操作"),
                    "Strategy_Detail": "策略細節",
                    "User_Advice": st.column_config.TextColumn("持單操作建議", help="針對左側設定的部位給出的建議"),
                    "User_Note": "持單細節"
                },
                hide_index=True
            )

            # --- B. 視覺化圖表 (移至下方) ---
            st.markdown("---")
            st.subheader("📊 近 60 根 K 棒走勢")
            
            df_chart = df_clean.tail(60)
            # 為了標記，我們需要對應的歷史紀錄
            df_hist_chart = df_history.tail(60)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart['Time'], y=df_chart['Close'], mode='lines+markers', name='Price', line=dict(color='#1f77b4', width=2)))
            
            # 標記自動策略的買賣點
            buys = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('買進')]
            sells = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('放空')]
            exits = df_hist_chart[df_hist_chart['Strategy_Action'].str.contains('出場')]
            
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys['Time'], y=buys['Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=15, color='red')))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells['Time'], y=sells['Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=15, color='green')))
            if not exits.empty:
                fig.add_trace(go.Scatter(x=exits['Time'], y=exits['Close'], mode='markers', name='Exit', marker=dict(symbol='x', size=12, color='orange')))
            
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis_type='category')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 請在左側貼上資料以開始")
