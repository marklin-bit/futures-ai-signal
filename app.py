import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os
import io

# 1. 網頁設定 (直接設定視窗標題，隱藏頁面內的大標題)
st.set_page_config(page_title="AI 交易訊號戰情室", layout="wide", initial_sidebar_state="expanded")

# CSS 美化 (縮減頂部空白，讓儀表板更緊湊)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
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
        """
        [防呆機制] 檢查時間是否為連續的 5 分鐘
        """
        if 'Time' not in df.columns:
            return [], "找不到時間欄位，無法檢查連續性。"
        
        try:
            # 嘗試轉換時間格式
            time_series = pd.to_datetime(df['Time'])
            # 計算時間差
            diffs = time_series.diff()
            
            # 找出間隔不等於 5 分鐘的列 (排除第一筆 NaN)
            # 5分鐘 = 300秒
            # 容許跨日 (例如 13:45 -> 隔日 08:45)，但盤中必須連續
            # 這裡做嚴格檢查：只要不是 5 分鐘就警示，使用者自行判斷是否為跨日
            discontinuous_indices = []
            
            for i in range(1, len(diffs)):
                delta = diffs.iloc[i]
                if delta.total_seconds() != 300: # 300秒 = 5分鐘
                    curr_time = time_series.iloc[i]
                    prev_time = time_series.iloc[i-1]
                    discontinuous_indices.append(f"{prev_time.strftime('%H:%M')} -> {curr_time.strftime('%H:%M')} (間隔 {delta})")
            
            return discontinuous_indices, None
            
        except Exception as e:
            return [], f"時間格式解析失敗: {e}"

    def process(self):
        if self.raw_df is None or self.raw_df.empty:
            return pd.DataFrame(), [], []

        df = self.raw_df.copy()
        df.columns = df.columns.astype(str)
        df.rename(columns=lambda x: x.replace('\n', '').strip(), inplace=True)
        
        clean_map = {}
        for col in df.columns:
            if col in self.rename_map:
                clean_map[col] = self.rename_map[col]
            else:
                for k, v in self.rename_map.items():
                    if k in col:
                        clean_map[col] = v
                        break
        df.rename(columns=clean_map, inplace=True)
        
        # 欄位檢查
        missing_features = []
        for col in self.feature_cols:
            if col not in df.columns:
                missing_features.append(col)
                df[col] = 0 
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 時間連續性檢查
        discontinuities = []
        if 'Time' in df.columns:
            discontinuities, err = self.validate_time_continuity(df)
        
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, missing_features, discontinuities

class StrategyEngine:
    def __init__(self, df, models, params):
        self.df = df
        self.models = models
        self.params = params

    def run_historical_review(self):
        position = 0 
        entry_price = 0.0
        entry_index = 0
        
        history_records = []
        X_all = self.df[DataProcessor(None).feature_cols]
        
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_features = X_all.iloc[[i]]
            current_close = current_bar.get('Close', 0)
            current_time = current_bar.get('Time', f"K_{i}")
            
            record = {
                'Index': i,
                'Time': current_time,
                'Close': current_close,
                'Suggestion': '觀望',
                'Confidence': 0.0,
                'Detail': ''
            }

            if position == 0:
                prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
                prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
                
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    position = 1
                    entry_price = current_close
                    entry_index = i
                    record['Suggestion'] = '🔥 買進'
                    record['Confidence'] = prob_long
                    record['Detail'] = f"做多 {prob_long:.0%}"
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    position = -1
                    entry_price = current_close
                    entry_index = i
                    record['Suggestion'] = '⚡ 放空'
                    record['Confidence'] = prob_short
                    record['Detail'] = f"做空 {prob_short:.0%}"
                else:
                    record['Detail'] = f"多:{prob_long:.2f}/空:{prob_short:.2f}"

            elif position == 1:
                floating_pnl = current_close - entry_price
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    record['Suggestion'] = '🛑 停損'
                    record['Detail'] = f"損 {floating_pnl}"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                    
                    exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        record['Suggestion'] = '🟢 出場'
                        record['Confidence'] = exit_prob
                        record['Detail'] = f"率 {exit_prob:.0%}"
                    else:
                        record['Suggestion'] = '續抱'
                        record['Detail'] = f"帳 {floating_pnl}"

            elif position == -1:
                floating_pnl = entry_price - current_close
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    record['Suggestion'] = '🛑 停損'
                    record['Detail'] = f"損 {floating_pnl}"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]

                    exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        record['Suggestion'] = '🔴 出場'
                        record['Confidence'] = exit_prob
                        record['Detail'] = f"率 {exit_prob:.0%}"
                    else:
                        record['Suggestion'] = '續抱'
                        record['Detail'] = f"帳 {floating_pnl}"
            
            history_records.append(record)
            
        return pd.DataFrame(history_records)

    def run_realtime_advice(self, user_position, entry_price, bars_held):
        last_idx = len(self.df) - 1
        current_features = self.df.iloc[[last_idx]][DataProcessor(None).feature_cols].copy()
        current_close = self.df.iloc[last_idx].get('Close', 0)
        
        advice = {"Action": "Wait", "Confidence": 0.0, "PnL": 0.0, "Message": "資料不足", "Type": "Neutral"}

        if user_position == 'Empty':
            prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
            prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
            
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
                advice['Confidence'] = exit_prob
                
                if exit_prob > self.params['exit_threshold']:
                    advice.update({"Action": "Exit", "Message": f"🚀 AI 建議多單出場 (機率 {exit_prob:.0%})", "Type": "Exit"})
                else:
                    advice.update({"Action": "Hold", "Message": f"⚓ AI 建議續抱 (出場率 {exit_prob:.0%})", "Type": "Hold"})

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
                advice['Confidence'] = exit_prob
                
                if exit_prob > self.params['exit_threshold']:
                    advice.update({"Action": "Exit", "Message": f"🚀 AI 建議空單出場 (機率 {exit_prob:.0%})", "Type": "Exit"})
                else:
                    advice.update({"Action": "Hold", "Message": f"⚓ AI 建議續抱 (出場率 {exit_prob:.0%})", "Type": "Hold"})

        return advice

# ==========================================
# 3. 載入模型
# ==========================================
@st.cache_resource
def load_models():
    model_names = ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']
    loaded_models = {}
    paths_to_try = ['', 'models/']
    for name in model_names:
        model = None
        for path in paths_to_try:
            file_path = f"{path}{name}.pkl"
            if os.path.exists(file_path):
                try: model = joblib.load(file_path); break
                except: pass
        if model: loaded_models[name] = model
        else: return None
    return loaded_models

# ==========================================
# 4. 網頁介面主邏輯 (儀表板佈局)
# ==========================================

# 建立兩欄佈局：左側輸入(30%)，右側儀表板(70%)
left_col, right_col = st.columns([1, 2.5])

models = load_models()

# --- 左側欄位：控制與輸入 ---
with left_col:
    st.subheader("🛠️ 數據與參數")
    
    # 參數設定區
    with st.expander("⚙️ 策略參數設定", expanded=False):
        entry_threshold = st.slider("進場信心", 0.5, 0.95, 0.80, 0.05)
        exit_threshold = st.slider("出場機率", 0.3, 0.9, 0.50, 0.05)
        hard_stop = st.number_input("硬停損點數", value=100, step=10)

    # 部位設定區
    st.markdown("##### 👤 目前真實部位")
    user_pos_type = st.radio("持倉狀態", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"], label_visibility="collapsed")
    user_entry_price = 0.0
    user_bars_held = 0
    if user_pos_type != "空手 (Empty)":
        c1, c2 = st.columns(2)
        user_entry_price = c1.number_input("成本", value=17500.0, step=1.0)
        user_bars_held = c2.number_input("K棒數", value=1, step=1, min_value=1)

    st.markdown("---")
    
    # 資料輸入區 (Tabs)
    tab1, tab2 = st.tabs(["📝 貼上資料", "🔄 即時串接"])
    
    df_input = None
    with tab1:
        st.caption("請從 Excel 複製含標題的數據 (時間, 收盤價, K, D, 布林, MA斜率...)")
        paste_data = st.text_area("Ctrl+V 貼上區", height=250, label_visibility="collapsed")
        if paste_data:
            try:
                df_input = pd.read_csv(io.StringIO(paste_data), sep='\t')
            except: st.error("格式錯誤")
    
    with tab2:
        st.info("🚧 此功能開發中\n\n未來將透過 API 自動抓取報價，實現全自動訊號推播。")

# --- 右側欄位：戰情儀表板 ---
with right_col:
    if models is None:
        st.error("⚠️ 模型載入失敗，請檢查 GitHub 檔案。")
    
    elif df_input is not None and not df_input.empty:
        processor = DataProcessor(df_input)
        df_clean, missing_cols, discontinuities = processor.process()
        
        # 1. 錯誤檢查
        if missing_cols:
            st.error(f"❌ 資料缺少關鍵欄位：{missing_cols}")
        else:
            if discontinuities:
                with st.expander(f"⚠️ 警告：偵測到 {len(discontinuities)} 處時間不連續", expanded=True):
                    st.warning("請確認這是否為跨日或休市，否則技術指標可能失真。")
                    st.write(discontinuities[:5]) # 只顯示前5個

            # 2. 執行策略
            params = {'entry_threshold': entry_threshold, 'exit_threshold': exit_threshold, 'hard_stop': hard_stop}
            engine = StrategyEngine(df_clean, models, params)
            
            # 取得即時建議
            pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
            advice = engine.run_realtime_advice(pos_map[user_pos_type], user_entry_price, user_bars_held)
            
            # 取得歷史建議
            df_history = engine.run_historical_review()
            last_bar = df_clean.iloc[-1]

            # --- A. 頂部關鍵數據卡片 ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📊 最新時間", str(last_bar.get('Time', 'N/A'))[-5:]) # 只顯示 HH:MM
            m2.metric("💰 收盤價", f"{last_bar.get('Close', 0):.0f}")
            
            # 根據建議類型變色
            delta_color = "off"
            if advice['Type'] in ['Buy', 'Exit']: delta_color = "normal" # 綠色/上升
            elif advice['Type'] in ['Sell', 'Stop']: delta_color = "inverse" # 紅色/下降
            
            m3.metric("🤖 AI 決策", advice['Type'], delta=advice['Message'], delta_color=delta_color)
            
            pnl_show = f"{advice['PnL']:.0f}" if user_pos_type != "空手 (Empty)" else "-"
            m4.metric("🎯 信心/損益", f"{advice['Confidence']:.0%}", delta=pnl_show)

            st.markdown("---")

            # --- B. 視覺化圖表 (K線 + 訊號) ---
            # 為了效能，只畫最後 60 根
            display_len = 60
            df_chart = df_clean.tail(display_len)
            df_hist_chart = df_history.tail(display_len)
            
            fig = go.Figure()
            # 價格線
            fig.add_trace(go.Scatter(x=df_chart['Time'], y=df_chart['Close'], mode='lines+markers', name='Price', line=dict(color='#1f77b4')))
            
            # 標記歷史上的買賣建議 (為了不讓圖太亂，只標進場點)
            buys = df_hist_chart[df_hist_chart['Suggestion'].str.contains('買進')]
            sells = df_hist_chart[df_hist_chart['Suggestion'].str.contains('放空')]
            
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys['Time'], y=buys['Close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=15, color='red')))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells['Time'], y=sells['Close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=15, color='green')))

            fig.update_layout(
                title="近 60 根 K 棒走勢與歷史訊號",
                margin=dict(l=0, r=0, t=30, b=0),
                height=350,
                xaxis_type='category' # 避免時間不連續產生的空白
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- C. 歷史建議表格 ---
            st.subheader("📜 歷史訊號回放 (倒序)")
            
            # 格式化表格，讓它更像看盤軟體的明細
            df_show = df_history[['Time', 'Close', 'Suggestion', 'Detail']].iloc[::-1] # 倒序
            
            # 使用 dataframe 的 column config 加上顏色條或圖示
            st.dataframe(
                df_show,
                use_container_width=True,
                height=300,
                column_config={
                    "Suggestion": st.column_config.TextColumn(
                        "AI 建議",
                        help="當時 AI 給出的操作建議",
                    ),
                    "Confidence": st.column_config.ProgressColumn(
                        "信心度",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )

    else:
        # 空白狀態的引導畫面
        st.info("👈 請先在左側貼上 Excel 資料以啟動戰情室")
        st.markdown("""
        ### 🚀 快速上手指南
        1. **複製資料**：從您的 Excel 或看盤軟體複製包含技術指標的數據。
        2. **貼上**：貼到左側的文字框中。
        3. **設定部位**：如果您手上已有單，請在左側設定，AI 會切換為「出場模式」。
        4. **看訊號**：右側儀表板會即時顯示最新建議。
        """)
