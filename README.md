# Linear Regression CRISP-DM 專案

**Demo: [https://bcwnbmzvux7ycgwzkhmfzh.streamlit.app/](https://bcwnbmzvux7ycgwzkhmfzh.streamlit.app/)**

本專案是一個基於 Streamlit 的單檔案應用程式，展示了如何遵循 CRISP-DM (Cross-Industry Standard Process for Data Mining) 工作流程來開發線性迴歸模型。

## 專案功能

- **合成資料生成**：可自定義樣本數、斜率、截距以及噪聲分布。
- **CRISP-DM 工作流程展示**：
  1. **業務理解**：定義預測目標。
  2. **資料理解**：原始資料預覽與敘述性統計。
  3. **資料準備**：訓練集/測試集分割與特徵標準化。
  4. **建模**：使用 scikit-learn 訓練線性迴歸模型。
  5. **評估**：計算 MSE、RMSE 與 R²，並視覺化迴歸結果。
  6. **部署**：即時預測功能與模型匯出。

## 技術棧

- Python 3
- Streamlit
- scikit-learn
- Pandas, NumPy
- Matplotlib
- joblib

## 如何執行

1. 安裝依賴項：
   ```bash
   pip install -r requirements.txt
   ```

2. 啟動 Streamlit 應用：
   ```bash
   streamlit run app.py
   ```
