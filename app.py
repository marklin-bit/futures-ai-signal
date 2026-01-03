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
        self.rename_map = {
            '布林通道寬度': 'Bandwidth',
            'MA斜率\n0平/1上/-1下': 'MA_Slope',
            'MA斜率': 'MA_Slope', # 容錯簡寫
            '布林帶寬度變化率': 'Bandwidth_Rate',
            '相對成交量': 'Rel_Volume',
            'K(36,3)': 'K',
            'K值': 'K', # 容錯
            'D(36,3)': 'D',
            'D值': 'D', # 容錯
            '收盤時\n通道位置': 'Position_in_Channel',
            '通道位置': 'Position_in_Channel',
            '波動率': 'Volatility',
            'K 棒\n相對強度': 'K_Strength',
            'K棒強度': 'K_Strength',
            '實體佔比': 'Body_Ratio',
            'Week': 'Week',
            '星期': 'Week',
            '結算日\n(0/1周結算/2月結算)': 'Settlement_Day',
            '結算日': 'Settlement_Day',
            '時段\n(0盤初/1盤中/2盤尾)': 'Time_Segment',
            '時段': 'Time_Segment',
            '單別\n1多單/2空單': 'Order_Type',
            '動作\n0無/1買進/2持單/3賣出': 'Action',
            '收盤價': 'Close', 
            '開盤價': 'Open',
            '最高價': 'High',
            '最低價': 'Low',
            '收盤時間': 'Time',
            '時間': 'Time'
        }
        self.exit_feature_cols = self.feature_cols + ['Floating_PnL', 'Hold_Bars']

    def process(self):
        # 避免空資料
        if self.raw_df is None or self.raw_df.empty:
            return pd.DataFrame()

        df = self.raw_df.copy()
        
        # 1. 欄位更名 (處理換行符號與簡寫)
        # 先把 columns 轉成字串以免出錯
        df.columns = df.columns.astype(str)
        df.rename(columns=lambda x: x.replace('\n', '').strip(), inplace=True)
        
        # 模糊比對清洗
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
        
        # 2. 強制轉型 (處理非數值雜訊)
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # 如果缺欄位，暫時補 0 並警告 (避免程式崩潰)
                df[col] = 0
        
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

class StrategyEngine:
    def __init__(self, df, models, params):
        self.df = df
        self.models = models
        self.params = params
        self.trades = []
        self.signals = [] 

    def run(self):
        position = 0 
        entry_price = 0.0
        entry_index = 0
        
        X_all = self.df[DataProcessor(None).feature_cols]
        
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_features = X_all.iloc[[i]]
            current_close = current_bar.get('Close', 0)
            current_time = current_bar.get('Time', f"K_{i}")
            
            signal_data = {
                'Time': current_time,
                'Close': current_close,
                'Action': 'Wait',
                'Position': position,
                'Prob_Long': 0.0,
                'Prob_Short': 0.0,
                'Prob_Exit': 0.0,
                'PnL': 0.0
            }

            # --- 策略邏輯 ---
            if position == 0:
                prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
                prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
                
                signal_data['Prob_Long'] = prob_long
                signal_data['Prob_Short'] = prob_short
                
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    position = 1
                    entry_price = current_close
                    entry_index = i
                    signal_data['Action'] = 'Buy'
                    self.trades.append({'Idx': i, 'Type': 'Long', 'Price': current_close, 'Time': current_time})
                
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    position = -1
                    entry_price = current_close
                    entry_index = i
                    signal_data['Action'] = 'Sell'
                    self.trades.append({'Idx': i, 'Type': 'Short', 'Price': current_close, 'Time': current_time})
                    
            elif position == 1:
                floating_pnl = current_close - entry_price
                signal_data['PnL'] = floating_pnl
                
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    signal_data['Action'] = 'StopLoss'
                    self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_col_order = DataProcessor(None).exit_feature_cols
                    exit_feats = exit_feats[exit_col_order]

                    exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    signal_data['Prob_Exit'] = exit_prob
                    
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        signal_data['Action'] = 'Exit'
                        self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})

            elif position == -1:
                floating_pnl = entry_price - current_close
                signal_data['PnL'] = floating_pnl
                
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    signal_data['Action'] = 'StopLoss'
                    self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})
                else:
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    exit_col_order = DataProcessor(None).exit_feature_cols
                    exit_feats = exit_feats[exit_col_order]

                    exit_prob = self.models['Short_Exit_Model'].predict_proba(exit_feats)[0][1]
                    signal_data['Prob_Exit'] = exit_prob
                    
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        signal_data['Action'] = 'Exit'
                        self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})
            
            signal_data['Position'] = position
            self.signals.append(signal_data)
            
        return pd.DataFrame(self.signals), pd.DataFrame(self.trades)

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
                except Exception as e:
                    st.error(f"讀取模型 {name} 失敗: {e}")
        if model:
            loaded_models[name] = model
        else:
            return None
    return loaded_models

# ==========================================
# 3. 網頁介面主邏輯
# ==========================================
st.title("🚀 台指期 5分K 四模型即時訊號站")

# 側邊欄
st.sidebar.header("⚙️ 策略參數")
entry_threshold = st.sidebar.slider("進場信心門檻 (Confidence)", 0.5, 0.95, 0.80, 0.05)
exit_threshold = st.sidebar.slider("出場機率門檻", 0.3, 0.9, 0.50, 0.05)
hard_stop = st.sidebar.number_input("硬停損點數 (Hard Stop)", value=100, step=10)

models = load_models()
if models is None:
    st.error("⚠️ 找不到模型檔案 (.pkl)。請確認 GitHub 上傳正確。")
    st.stop()
else:
    st.sidebar.success("✅ 模型載入成功")

# --- 資料輸入區塊 ---
st.subheader("📋 資料輸入")
tab1, tab2 = st.tabs(["📝 貼上 Excel 資料", "📂 上傳 CSV 檔案"])

df_input = None

with tab1:
    st.info("請從 Excel 選取包含標題的資料範圍，複製並貼在下方 (包含收盤價、K、D、布林寬度等欄位)。")
    paste_data = st.text_area("在此貼上資料 (Ctrl+V):", height=150)
    if paste_data:
        try:
            # 自動辨識 Tab 分隔 (Excel 預設)
            df_input = pd.read_csv(io.StringIO(paste_data), sep='\t')
        except Exception as e:
            st.error(f"資料解析失敗: {e}")

with tab2:
    uploaded_file = st.file_uploader("上傳 CSV 檔案", type=['csv'])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV 讀取失敗: {e}")

# --- 執行分析 ---
if df_input is not None and not df_input.empty:
    try:
        # 清洗與處理
        processor = DataProcessor(df_input)
        df_clean = processor.process()
        
        # 檢查必要特徵是否存在
        missing_cols = [col for col in processor.feature_cols if col not in df_clean.columns]
        if missing_cols:
            # 如果缺欄位，上面 process 已經補 0，但我們可以提醒一下使用者
            st.caption(f"注意：部分欄位未偵測到，已自動補 0: {missing_cols[:3]}...")
        
        # 執行引擎
        params = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'hard_stop': hard_stop
        }
        engine = StrategyEngine(df_clean, models, params)
        df_signals, df_trades = engine.run()
        
        # --- 顯示最新狀態 (Dashboard) ---
        last_row = df_signals.iloc[-1]
        st.markdown("---")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新時間", str(last_row['Time']))
        c2.metric("收盤價", f"{last_row['Close']:.0f}")
        
        # 狀態燈號
        status_text = "觀望 (Wait)"
        if last_row['Position'] == 1:
            status_text = "🟢 持有多單"
            if last_row['Prob_Exit'] > exit_threshold: status_text += " (建議出場!)"
        elif last_row['Position'] == -1:
            status_text = "🔴 持有空單"
            if last_row['Prob_Exit'] > exit_threshold: status_text += " (建議出場!)"
        elif last_row['Action'] == 'Buy':
            status_text = "🔥 買進訊號"
        elif last_row['Action'] == 'Sell':
            status_text = "⚡ 放空訊號"
        
        c3.metric("AI 建議", status_text)
        
        # 信心度
        conf = max(last_row['Prob_Long'], last_row['Prob_Short']) if last_row['Position'] == 0 else last_row['Prob_Exit']
        label = "進場信心" if last_row['Position'] == 0 else "出場機率"
        c4.metric(label, f"{conf:.1%}")

        # --- 圖表區 (如果有超過 5 筆資料才畫圖，避免太醜) ---
        if len(df_signals) > 5:
            st.subheader("📊 走勢回測圖")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['Close'], mode='lines', name='Close', line=dict(color='gray')))
            
            # 標記買賣點
            buys = df_signals[df_signals['Action'] == 'Buy']
            sells = df_signals[df_signals['Action'] == 'Sell']
            exits = df_signals[df_signals['Action'].isin(['Exit', 'StopLoss'])]
            
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=12, color='green')))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=12, color='red')))
            if not exits.empty:
                fig.add_trace(go.Scatter(x=exits.index, y=exits['Close'], mode='markers', name='Exit', marker=dict(symbol='x', size=10, color='orange')))
            
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("查看原始數據與訊號"):
            st.dataframe(df_signals.tail(10))

    except Exception as e:
        st.error(f"運算發生錯誤: {e}")
else:
    st.info("👋 請在上方貼上 Excel 資料 (包含標題列) 以開始分析。")
