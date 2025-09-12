# Roadmap: Final Migration to Binary Trade/No-Trade Model

Goal:  
Complete the transition from the old three-class (1 = long, 0 = hold, –1 = short) model to a **binary trade/no-trade** architecture, while finishing the remaining unimplemented tasks and ensuring smooth integration with the existing Regime Detector and Strategy Selector.

---

## 1. Label Redefinition  **(Critical – Data Layer)**  
**Description:**  
Create a new binary target that reflects only the decision to trade or skip, with strict prevention of look-ahead bias.

**Instructions:**  
- **Define the signal horizon and profit threshold** (e.g., forward return over N bars > +τ after fees → label 1, otherwise 0).  
- Ensure every feature used for training is available at or before the decision timestamp.  
- Add the new column `label_binary` to the master dataset and update all ETL/loaders to use it.  
- Version the dataset (e.g., `v2`) to keep a clean history of the old vs. new labeling.

---

## 2. Binary Model Configuration  **(Critical – ML Core)**  
**Description:**  
Reconfigure the ML training pipeline to handle a binary target while re-using the existing feature-engineering and cross-validation infrastructure.

**Instructions:**  
- Update model definitions to output a single probability `p_trade` (e.g., logistic/gradient boosting).  
- Apply class weighting if the trade/no-trade ratio is imbalanced.  
- Use **walk-forward validation**, not random splits, to respect time-series order.  
- Track both standard metrics (AUC, F1) and economic metrics (expected PnL, Sharpe).

---

## 3. Probability Calibration & Threshold Optimisation  **(Performance Layer)**  
**Description:**  
Ensure probability outputs are well-calibrated and choose a trade threshold that maximizes profit after costs.

**Instructions:**  
- Calibrate probabilities with Platt scaling or isotonic regression on validation folds.  
- Grid-search thresholds (e.g., 0.5–0.9) and evaluate expected PnL, Sharpe, and max drawdown.  
- Store the selected threshold in configuration so it can be adjusted without retraining.

---

## 4. Integration with Regime Detector & Strategy Selector  **(Decision Layer)**  
**Description:**  
Connect the binary entry model to the existing modules that already determine trade direction and strategy.

**Instructions:**  
- When `p_trade > threshold`, trigger the **Strategy Selector**, which uses the **Regime Detector** to choose long/short logic.  
- Verify that the output from the selector (direction and chosen strategy) flows to the Risk Manager and Order Executor.  
- Add integration tests that feed market data → binary model → selector → executor and assert that orders are correct.

---

## 5. Monitoring & Alerting Enhancements  **(Operations Layer)**  
**Description:**  
Extend the current Prometheus metrics and trade logs to track the new binary model’s health and calibration.

**Instructions:**  
- Expose metrics: current threshold, average `p_trade`, number of trades per regime, realized vs. predicted hit rate.  
- Add alert rules for drift detection (e.g., sudden change in trade frequency or calibration error).  
- Ensure logs capture `p_trade`, regime, selected strategy, and final order details for every decision.

---

## 6. Documentation Update  **(Knowledge & Maintenance)**  
**Description:**  
Provide complete, current documentation of the binary migration and new operational flow.

**Instructions:**  
- Update README or internal wiki to explain:
  - Motivation for binary trade/no-trade migration.
  - Labeling rules and horizon.
  - Calibration and threshold selection process.
  - Full decision flow: **Binary Entry Model → Strategy Selector (Regime Detector) → Risk Manager → Order Executor**.
- Include diagrams and a sample configuration file for quick onboarding of future contributors.

---

## Execution Order
1. **Label Redefinition**  
2. **Binary Model Configuration**  
3. **Probability Calibration & Threshold Optimisation**  
4. **Integration with Regime Detector & Strategy Selector**  
5. **Monitoring & Alerting Enhancements**  
6. **Documentation Update**

---

**Outcome:**  
After completing these steps the framework will operate on a clean binary entry model, with calibrated probabilities, seamless direction control via the existing regime/strategy modules, robust monitoring, and clear documentation—ready for full backtesting and live deployment.