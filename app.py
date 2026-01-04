import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os
import io

# 設定網頁標題與寬度
st.set_page_config(page_title="台指期 AI 交易訊號站", layout="wide")

# ==========================================
# 1. 核心類別定義
# ==========================================
class DataProcessor:
    def __init__(self, df):
        self.raw_df = df
        # 定義特徵欄位 (必須與訓練時完全一樣)
        self.feature_cols = [
            'Bandwidth', 'MA_Slope', 'Bandwidth_Rate', 'Rel_Volume',
            'K', 'D', 'Position_in_Channel', 'Volatility', 
            'K_Strength', 'Body_Ratio', 'Week', 'Settlement_Day', 'Time_Segment'
        ]
        # 定義中文對照 (方便使用者上傳原始檔)
        # 鍵值(Key)是使用者Excel可能的欄位名，值(Value)是程式內部用的英文名
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

    def process(self):
        if self.raw_df is None or self.raw_df.empty:
            return pd.DataFrame(), []

        df = self.raw_df.copy()
        
        # 1. 欄位更名
        # 先轉字串處理換行
        df.columns = df.columns.astype(str)
        df.rename(columns=lambda x: x.replace('\n', '').strip(), inplace=True)
        
        clean_map = {}
        for col in df.columns:
            # 嘗試完全比對
            if col in self.rename_map:
                clean_map[col] = self.rename_map[col]
            else:
                # 嘗試部分比對 (例如 "MA斜率" in "MA斜率\n0平...")
                for k, v in self.rename_map.items():
                    if k in col:
                        clean_map[col] = v
                        break
        df.rename(columns=clean_map, inplace=True)
        
        # 2. 檢查是否有缺漏的關鍵欄位
        missing_features = []
        for col in self.feature_cols:
            if col not in df.columns:
                missing_features.append(col)
                df[col] = 0 # 暫時補0防崩潰，但會回傳缺失清單
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, missing_features

class StrategyEngine:
    def __init__(self, df, models, params):
        self.df = df
        self.models = models
        self.params = params

    def run_historical_review(self):
        """
        模擬從第一筆資料開始跑到最後一筆 (歷史回放)
        """
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

            # 模擬策略
            if position == 0:
                prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
                prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
                
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    position = 1
                    entry_price = current_close
                    entry_index = i
                    record['Suggestion'] = '🔥 買進'
                    record['Confidence'] = prob_long
                    record['Detail'] = f"做多信心 {prob_long:.0%}"
                
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    position = -1
                    entry_price = current_close
                    entry_index = i
                    record['Suggestion'] = '⚡ 放空'
                    record['Confidence'] = prob_short
                    record['Detail'] = f"做空信心 {prob_short:.0%}"
                else:
                    record['Detail'] = f"多:{prob_long:.2f} / 空:{prob_short:.2f}"

            elif position == 1: # 模擬持多
                floating_pnl = current_close - entry_price
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    record['Suggestion'] = '🛑 停損出場'
                    record['Detail'] = f"虧損 {floating_pnl} 點"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                    
                    exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        record['Suggestion'] = '🟢 多單出場'
                        record['Confidence'] = exit_prob
                        record['Detail'] = f"出場機率 {exit_prob:.0%}"
                    else:
                        record['Suggestion'] = '持多續抱'
                        record['Detail'] = f"帳面 {floating_pnl} 點"

            elif position == -1: # 模擬持空
                floating_pnl = entry_price - current_close
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    record['Suggestion'] = '🛑 停損出場'
                    record['Detail'] = f"虧損 {floating_pnl} 點"
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]

                    exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        record['Suggestion'] = '🔴 空單出場'
                        record['Confidence'] = exit_prob
                        record['Detail'] = f"出場機率 {exit_prob:.0%}"
                    else:
                        record['Suggestion'] = '持空續抱'
                        record['Detail'] = f"帳面 {floating_pnl} 點"
            
            history_records.append(record)
            
        return pd.DataFrame(history_records)

    def run_realtime_advice(self, user_position, entry_price, bars_held):
        """
        針對「最後一筆資料」，結合「使用者真實部位」給出建議
        """
        # 取最後一筆
        last_idx = len(self.df) - 1
        current_features = self.df.iloc[[last_idx]][DataProcessor(None).feature_cols].copy()
        current_close = self.df.iloc[last_idx].get('Close', 0)
        
        advice = {
            "Action": "Wait",
            "Confidence": 0.0,
            "PnL": 0.0,
            "Message": "資料不足"
        }

        # 1. 如果使用者是空手 (Empty) -> 跑進場模型
        if user_position == 'Empty':
            prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
            prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
            
            if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                advice['Action'] = "Buy"
                advice['Confidence'] = prob_long
                advice['Message'] = "多方訊號強勢，建議買進"
            elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                advice['Action'] = "Sell"
                advice['Confidence'] = prob_short
                advice['Message'] = "空方訊號強勢，建議放空"
            else:
                advice['Action'] = "Wait"
                advice['Confidence'] = max(prob_long, prob_short)
                advice['Message'] = f"訊號不明確 (多:{prob_long:.2f} / 空:{prob_short:.2f})"

        # 2. 如果使用者持有多單 (Long) -> 跑多單出場模型
        elif user_position == 'Long':
            floating_pnl = current_close - entry_price
            advice['PnL'] = floating_pnl
            
            # 硬停損檢查
            if floating_pnl <= -self.params['hard_stop']:
                advice['Action'] = "StopLoss"
                advice['Confidence'] = 1.0
                advice['Message'] = f"觸發硬停損 (-{self.params['hard_stop']}點)，請立即出場"
            else:
                # 準備特徵
                exit_feats = current_features.copy()
                exit_feats['Floating_PnL'] = floating_pnl
                exit_feats['Hold_Bars'] = bars_held
                exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                
                exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                advice['Confidence'] = exit_prob
                
                if exit_prob > self.params['exit_threshold']:
                    advice['Action'] = "Exit"
                    advice['Message'] = f"AI 建議多單出場 (機率 {exit_prob:.0%})"
                else:
                    advice['Action'] = "Hold"
                    advice['Message'] = f"AI 建議續抱 (出場率僅 {exit_prob:.0%})"

        # 3. 如果使用者持有空單 (Short) -> 跑空單出場模型
        elif user_position == 'Short':
            floating_pnl = entry_price - current_close
            advice['PnL'] = floating_pnl
            
            if floating_pnl <= -self.params['hard_stop']:
                advice['Action'] = "StopLoss"
                advice['Confidence'] = 1.0
                advice['Message'] = f"觸發硬停損 (-{self.params['hard_stop']}點)，請立即出場"
            else:
                exit_feats = current_features.copy()
                exit_feats['Floating_PnL'] = floating_pnl
                exit_feats['Hold_Bars'] = bars_held
                exit_feats = exit_feats[DataProcessor(None).exit_feature_cols]
                
                exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                advice['Confidence'] = exit_prob
                
                if exit_prob > self.params['exit_threshold']:
                    advice['Action'] = "Exit"
                    advice['Message'] = f"AI 建議空單出場 (機率 {exit_prob:.0%})"
                else:
                    advice['Action'] = "Hold"
                    advice['Message'] = f"AI 建議續抱 (出場率僅 {exit_prob:.0%})"

        return advice

# ==========================================
# 2. 載入模型
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
                try:
                    model = joblib.load(file_path)
                    break
                except: pass
        if model: loaded_models[name] = model
        else: return None
    return loaded_models

# ==========================================
# 3. 網頁介面主邏輯
# ==========================================
st.title("🚀 台指期 5分K 四模型即時訊號站")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 策略參數")
entry_threshold = st.sidebar.slider("進場信心門檻", 0.5, 0.95, 0.80, 0.05)
exit_threshold = st.sidebar.slider("出場機率門檻", 0.3, 0.9, 0.50, 0.05)
hard_stop = st.sidebar.number_input("硬停損點數", value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.header("👤 實戰部位設定")
st.sidebar.info("請在此輸入您目前的真實部位，AI 才能提供正確的出場建議。")
user_pos_type = st.sidebar.radio("目前持倉狀態", ["空手 (Empty)", "多單 (Long)", "空單 (Short)"])

user_entry_price = 0.0
user_bars_held = 0

if user_pos_type != "空手 (Empty)":
    user_entry_price = st.sidebar.number_input("進場成本價", value=17500.0, step=1.0)
    user_bars_held = st.sidebar.number_input("已持有 K 棒數", value=1, step=1, min_value=1)

# 載入模型
models = load_models()
if models is None:
    st.error("⚠️ 找不到模型檔案 (.pkl)。")
    st.stop()

# --- 資料輸入區塊 ---
st.subheader("📋 資料輸入")
st.info("💡 提示：Excel 複製時，請務必包含以下「關鍵欄位標題」(順序不拘)：\n"
        "收盤時間, 收盤價, K值, D值, 布林通道寬度, MA斜率, "
        "相對成交量, 通道位置, 波動率, K棒強度, 實體佔比, 星期, 結算日, 時段")

tab1, tab2 = st.tabs(["📝 貼上 Excel 資料", "📂 上傳 CSV 檔案"])

df_input = None
with tab1:
    st.caption("請從 Excel 複製資料 (含標題) 貼上。")
    paste_data = st.text_area("貼上區 (Ctrl+V):", height=150)
    if paste_data:
        try:
            df_input = pd.read_csv(io.StringIO(paste_data), sep='\t')
        except: st.error("資料解析失敗")

with tab2:
    uploaded_file = st.file_uploader("上傳 CSV", type=['csv'])
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
        except: st.error("讀取失敗")

# --- 執行分析 ---
if df_input is not None and not df_input.empty:
    processor = DataProcessor(df_input)
    # process 現在會回傳兩個值：資料表 和 缺失欄位清單
    df_clean, missing_cols = processor.process()
    
    # 檢查是否有缺失欄位，並發出警告
    if missing_cols:
        st.error(f"❌ 嚴重警告：偵測到資料缺少以下關鍵欄位，模型將無法正確運作！\n"
                 f"缺失欄位: {missing_cols}")
        st.stop() # 強制停止，避免算出錯誤數據
    
    params = {'entry_threshold': entry_threshold, 'exit_threshold': exit_threshold, 'hard_stop': hard_stop}
    engine = StrategyEngine(df_clean, models, params)

    # 1. 取得即時建議
    pos_map = {"空手 (Empty)": "Empty", "多單 (Long)": "Long", "空單 (Short)": "Short"}
    realtime_advice = engine.run_realtime_advice(pos_map[user_pos_type], user_entry_price, user_bars_held)

    # 2. 取得歷史回放
    df_history = engine.run_historical_review()

    # --- Dashboard 顯示 ---
    st.markdown("---")
    last_row = df_clean.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新時間", str(last_row.get('Time', 'N/A')))
    col2.metric("收盤價", f"{last_row.get('Close', 0):.0f}")
    
    # AI 建議燈號
    advice_color = "off"
    if realtime_advice['Action'] in ['Buy', 'Exit', 'StopLoss']: advice_color = "inverse"
    
    col3.metric("AI 實戰建議", realtime_advice['Message'])
    col4.metric("信心/機率", f"{realtime_advice['Confidence']:.1%}", delta=f"損益: {realtime_advice['PnL']:.0f}" if user_pos_type != "空手 (Empty)" else None)

    # --- 歷史建議清單 ---
    st.subheader("📜 歷史訊號回放列表")
    st.caption("以下列表展示：如果 AI 從第一筆資料就開始看盤，它會在每個時間點給出什麼建議？(這能幫您補回錯過的行情判斷)")
    
    # 整理表格顯示
    display_cols = ['Time', 'Close', 'Suggestion', 'Detail']
    # 把最新的排在最上面
    st.dataframe(df_history[display_cols].iloc[::-1], use_container_width=True)

else:
    st.info("👋 等待資料輸入中...")
