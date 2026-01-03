import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os

# 設定網頁標題與寬度
st.set_page_config(page_title="台指期 AI 交易訊號站", layout="wide")

# ==========================================
# 1. 核心類別定義 (必須與訓練時一致)
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
            '布林帶寬度變化率': 'Bandwidth_Rate',
            '相對成交量': 'Rel_Volume',
            'K(36,3)': 'K',
            'D(36,3)': 'D',
            '收盤時\n通道位置': 'Position_in_Channel',
            '波動率': 'Volatility',
            'K 棒\n相對強度': 'K_Strength',
            '實體佔比': 'Body_Ratio',
            'Week': 'Week',
            '結算日\n(0/1周結算/2月結算)': 'Settlement_Day',
            '時段\n(0盤初/1盤中/2盤尾)': 'Time_Segment',
            '單別\n1多單/2空單': 'Order_Type',
            '動作\n0無/1買進/2持單/3賣出': 'Action',
            '收盤價': 'Close', 
            '開盤價': 'Open',
            '最高價': 'High',
            '最低價': 'Low',
            '收盤時間': 'Time'
        }
        self.exit_feature_cols = self.feature_cols + ['Floating_PnL', 'Hold_Bars']

    def process(self):
        df = self.raw_df.copy()
        
        # 1. 欄位更名
        df.rename(columns=lambda x: x.replace('\n', '') if isinstance(x, str) else x, inplace=True)
        clean_map = {k.replace('\n', ''): v for k, v in self.rename_map.items()}
        df.rename(columns=clean_map, inplace=True)
        
        # 2. 強制轉型 (處理非數值雜訊)
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. 確保必要欄位存在 (若使用者上傳的只有 OHLC，這邊會過濾掉，導致無法預測)
        # 為了容錯，我們填補缺失值為 0 (但建議使用者上傳完整計算過指標的 CSV)
        df.dropna(subset=self.feature_cols, how='any', inplace=True)
        df.fillna(0, inplace=True)
        
        # 重置 index 以便後續回測迴圈使用
        df.reset_index(drop=True, inplace=True)
        return df

class StrategyEngine:
    def __init__(self, df, models, params):
        self.df = df
        self.models = models
        self.params = params
        self.trades = []
        self.signals = [] # 紀錄每一根 K 棒的狀態

    def run(self):
        position = 0 # 0:Empty, 1:Long, -1:Short
        entry_price = 0.0
        entry_index = 0
        entry_prob = 0.0
        
        X_all = self.df[DataProcessor(None).feature_cols]
        
        # 逐行掃描
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_features = X_all.iloc[[i]]
            current_close = current_bar.get('Close', 0)
            current_time = current_bar.get('Time', f"Bar_{i}")
            
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
                # 1. 空手狀態
                prob_long = self.models['Long_Entry_Model'].predict_proba(current_features)[0][1]
                prob_short = self.models['Short_Entry_Model'].predict_proba(current_features)[0][1]
                
                signal_data['Prob_Long'] = prob_long
                signal_data['Prob_Short'] = prob_short
                
                if prob_long > self.params['entry_threshold'] and prob_long > prob_short:
                    position = 1
                    entry_price = current_close
                    entry_index = i
                    entry_prob = prob_long
                    signal_data['Action'] = 'Buy'
                    self.trades.append({'Idx': i, 'Type': 'Long', 'Price': current_close, 'Time': current_time})
                
                elif prob_short > self.params['entry_threshold'] and prob_short > prob_long:
                    position = -1
                    entry_price = current_close
                    entry_index = i
                    entry_prob = prob_short
                    signal_data['Action'] = 'Sell'
                    self.trades.append({'Idx': i, 'Type': 'Short', 'Price': current_close, 'Time': current_time})
                    
            elif position == 1:
                # 2. 持有多單
                floating_pnl = current_close - entry_price
                signal_data['PnL'] = floating_pnl
                
                # 硬停損
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    signal_data['Action'] = 'StopLoss'
                    self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})
                else:
                    # AI 出場
                    exit_feats = current_features.copy()
                    exit_feats['Floating_PnL'] = floating_pnl
                    exit_feats['Hold_Bars'] = i - entry_index
                    # 確保特徵順序與訓練時一致
                    exit_col_order = DataProcessor(None).exit_feature_cols
                    exit_feats = exit_feats[exit_col_order]

                    exit_prob = self.models['Long_Exit_Model'].predict_proba(exit_feats)[0][1]
                    signal_data['Prob_Exit'] = exit_prob
                    
                    if exit_prob > self.params['exit_threshold']:
                        position = 0
                        signal_data['Action'] = 'Exit'
                        self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})

            elif position == -1:
                # 3. 持有空單
                floating_pnl = entry_price - current_close
                signal_data['PnL'] = floating_pnl
                
                # 硬停損
                if floating_pnl <= -self.params['hard_stop']:
                    position = 0
                    signal_data['Action'] = 'StopLoss'
                    self.trades.append({'Idx': i, 'Type': 'Exit', 'Price': current_close, 'PnL': floating_pnl, 'Time': current_time})
                else:
                    # AI 出場
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
# 2. 載入模型 (快取)
# ==========================================
@st.cache_resource
def load_models():
    # 請確保 .pkl 檔案與 app.py 在同一層目錄，或在 models/ 資料夾下
    model_names = ['Long_Entry_Model', 'Short_Entry_Model', 'Long_Exit_Model', 'Short_Exit_Model']
    loaded_models = {}
    
    # 嘗試兩種路徑 (根目錄 或 models/ 子目錄)
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
            return None # 只要有一個模型讀不到就回傳 None
            
    return loaded_models

# ==========================================
# 3. 網頁介面主邏輯
# ==========================================
st.title("🚀 台指期 5分K 四模型即時訊號站")

# 側邊欄：參數設定
st.sidebar.header("⚙️ 策略參數")
entry_threshold = st.sidebar.slider("進場信心門檻 (Confidence)", 0.5, 0.95, 0.80, 0.05)
exit_threshold = st.sidebar.slider("出場機率門檻", 0.3, 0.9, 0.50, 0.05)
hard_stop = st.sidebar.number_input("硬停損點數 (Hard Stop)", value=100, step=10)

# 載入模型
models = load_models()
if models is None:
    st.error("⚠️ 找不到模型檔案 (.pkl)。請將 'Long_Entry_Model.pkl' 等 4 個檔案上傳到 GitHub Repository 的根目錄或 'models/' 資料夾中。")
    st.stop()
else:
    st.sidebar.success("✅ 模型載入成功")

# 上傳檔案
uploaded_file = st.file_uploader("📂 上傳含有技術指標的 CSV 檔案 (至少60列)", type=['csv'])

if uploaded_file is not None:
    try:
        # 讀取並清洗資料
        df_raw = pd.read_csv(uploaded_file)
        processor = DataProcessor(df_raw)
        df_clean = processor.process()
        
        if len(df_clean) < 60:
            st.warning("⚠️ 資料筆數少於 60 筆，技術指標可能不穩定，但模型仍會嘗試運算。")
        
        # 執行策略引擎
        params = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'hard_stop': hard_stop
        }
        engine = StrategyEngine(df_clean, models, params)
        df_signals, df_trades = engine.run()
        
        # --- 顯示最新狀態 (最重要的即時訊號) ---
        last_row = df_signals.iloc[-1]
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最新時間", str(last_row['Time']))
        
        with col2:
            price_color = "normal"
            st.metric("收盤價", f"{last_row['Close']:.0f}")
            
        with col3:
            # 顯示當前建議
            status_text = "觀望 (Wait)"
            status_color = "off"
            if last_row['Position'] == 1:
                status_text = "🟢 持有多單"
                if last_row['Prob_Exit'] > exit_threshold:
                    status_text += " (建議出場!)"
            elif last_row['Position'] == -1:
                status_text = "🔴 持有空單"
                if last_row['Prob_Exit'] > exit_threshold:
                    status_text += " (建議出場!)"
            elif last_row['Action'] == 'Buy':
                status_text = "🔥 買進訊號 (Buy)"
            elif last_row['Action'] == 'Sell':
                status_text = "⚡ 放空訊號 (Sell)"
            
            st.metric("AI 建議", status_text)
            
        with col4:
            # 顯示信心度
            conf = 0.0
            if last_row['Position'] == 0:
                # 沒部位時看進場信心
                conf = max(last_row['Prob_Long'], last_row['Prob_Short'])
                label = "進場信心"
            else:
                # 有部位時看出場機率
                conf = last_row['Prob_Exit']
                label = "出場機率"
            
            st.metric(label, f"{conf:.1%}")

        # --- 繪製 K 線圖與買賣點 ---
        st.subheader("📊 訊號回測圖表")
        
        fig = go.Figure()
        
        # 價格線
        fig.add_trace(go.Scatter(
            x=df_signals.index, 
            y=df_signals['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='gray', width=1)
        ))
        
        # 買點 (Buy)
        buys = df_signals[df_signals['Action'] == 'Buy']
        fig.add_trace(go.Scatter(
            x=buys.index, 
            y=buys['Close'],
            mode='markers',
            name='Buy',
            marker=dict(symbol='triangle-up', size=12, color='green')
        ))
        
        # 賣點 (Sell)
        sells = df_signals[df_signals['Action'] == 'Sell']
        fig.add_trace(go.Scatter(
            x=sells.index, 
            y=sells['Close'],
            mode='markers',
            name='Sell',
            marker=dict(symbol='triangle-down', size=12, color='red')
        ))
        
        # 出場點 (Exit/StopLoss)
        exits = df_signals[df_signals['Action'].isin(['Exit', 'StopLoss'])]
        fig.add_trace(go.Scatter(
            x=exits.index, 
            y=exits['Close'],
            mode='markers',
            name='Exit',
            marker=dict(symbol='x', size=10, color='orange')
        ))

        fig.update_layout(height=500, xaxis_title="K棒序號 (Index)", yaxis_title="價格")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 顯示詳細資料表 ---
        with st.expander("查看詳細訊號數據"):
            st.dataframe(df_signals.tail(20)) # 只顯示最後 20 筆
            
    except Exception as e:
        st.error(f"處理檔案時發生錯誤: {e}")
        st.write("請確認上傳的 CSV 格式與訓練資料一致 (包含技術指標欄位)。")

else:
    st.info("👋 請從左側上傳 CSV 檔案以開始分析。")