## Test 1 2/5/2025

**Model & Specs**  
- **Model:** MLP classifier (scikit-learn)  
- **Features:** 18  
- **Neurons:** 50 & 100  
- **Activation:** relu  
- **Solver:** adam & adaptable  
- **Max Iterations:** 500  
- **Alpha:** 0.0001  

**Features List:**  
- Pct_Change_Close  
- Pct_Change_3d  
- Ma5  
- Ma20  
- Close_to_Ma5  
- Close_to_Ma20  
- Ma5_minus_Ma20  
- Range_Ratio  
- Pct_Change_Volume  
- Va5  
- Va20  
- Lagged_Volume_3d  
- Std_5d  
- Previous_Close  
- TR  
- ATR_14  
- RSI_14  
- Momentum_10  

---

### Sector Accuracy

The table below shows the overall (total) accuracy and, for each sector, the per-class “accuracy” (using class recall as a proxy) for:  
- **-1:** Loss  
- **0:** Break-even-ish  
- **1:** Profit

| Sector                  | Total Accuracy | Accuracy (-1) | Accuracy (0) | Accuracy (1) |
|-------------------------|----------------|---------------|--------------|--------------|
| Basic Materials         | 0.478          | 0.61          | 0.25         | 0.39         |
| Communication Services  | 0.488          | 0.52          | 0.16         | 0.50         |
| Consumer Cyclical       | 0.485          | 0.36          | 0.19         | 0.67         |
| Consumer Defensive      | 0.485          | 0.32          | 0.22         | 0.70         |
| Energy                  | 0.480          | 0.23          | 0.25         | 0.77         |
| Financial Services      | 0.488          | 0.22          | 0.24         | 0.80         |
| Healthcare              | 0.488          | 0.61          | 0.12         | 0.43         |
| Industrials             | 0.477          | 0.33          | 0.26         | 0.67         |
| Real Estate             | 0.494          | 0.32          | 0.31         | 0.69         |
| Technology              | 0.488          | 0.35          | 0.16         | 0.69         |
| Utilities               | 0.471          | 0.29          | 0.19         | 0.72         |
| Other                   | 0.476          | 0.48          | 0.48         | 0.48         |
---

