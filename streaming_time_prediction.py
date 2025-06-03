"""
Daily Listening Time Prediction Pipeline
=========================================
0. Baseline mean regressor
1.Linear models
   1.1 Multiple Linear Regression
   1.2 Ridge Regression
   1.3 Lasso Regression
   1.4 Polynomial Regression (interaction & quadratic terms)
   1.5 Feature dropped Linear Regression
2. Neural Network (MLPRegressor) – grid search over activations / hidden layers
3. Decision Tree Regressor (CART)
4. Ensemble models
   4.1 Bagging Regressor
   4.2 Random Forest Regressor
   4.3 AdaBoost Regressor
5. Gradient-Boosting 家族
   5.1 GradientBoostingRegressor      (sklearn)
   5.2 HistGradientBoostingRegressor  (sklearn ≥ 0.22)
   5.3 XGBRegressor                   (xgboost)
   5.4 LGBMRegressor                  (lightgbm)
   5.5 ExtraTreesRegressor            (sklearn)
"""

import matplotlib
matplotlib.use("Agg")               
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import warnings, math, joblib, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from packaging import version        
import sklearn
from sklearn.compose         import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import (StandardScaler, OneHotEncoder,
                                     PolynomialFeatures, MinMaxScaler)
from sklearn.linear_model    import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble        import (BaggingRegressor, RandomForestRegressor,
                                     AdaBoostRegressor, GradientBoostingRegressor,
                                     ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

warnings.filterwarnings("ignore")

#  2. 路径 
DATA_PATH  = Path("D:\\Desktop\\机器学习\\project\\Global_Music_Streaming_Listener_Preferences.csv")
PLOTS_DIR  = Path("plots");  PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)

# 3. 读取数据
raw    = pd.read_csv(DATA_PATH)
TARGET = "Minutes Streamed Per Day"
DROP   = ["User_ID"]               

y = raw[TARGET]
X = raw.drop(columns=[TARGET] + DROP)

# 4. 特征列 
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_tf = StandardScaler()

# 5. 预处理流水线
preproc_base = ColumnTransformer(
    [("num", num_tf, num_cols),
     ("cat", cat_tf, cat_cols)],
    sparse_threshold=0.0         # *确保整体 dense*
)

poly_preproc = ColumnTransformer(
    [("num_poly", Pipeline([("sc", StandardScaler()),
                            ("poly", PolynomialFeatures(2, include_bias=False))]), num_cols),
     ("cat", cat_tf, cat_cols)],
    sparse_threshold=0.0
)

# 6. 划分数据
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=pd.qcut(y, 10, duplicates="drop"))

# 7. 目标缩放器
def scaled(regressor):
    return TransformedTargetRegressor(regressor=regressor,
                                      transformer=StandardScaler())

# 8. 模型库 
αs = np.logspace(-3, 3, 25)
models = {
    # —— 线性家族 ——
    "OLS":           Pipeline([("prep", preproc_base), ("lr",   LinearRegression())]),
    "Ridge":         Pipeline([("prep", preproc_base), ("ridge", RidgeCV(alphas=αs))]),
    "Lasso":         Pipeline([("prep", preproc_base), ("lasso", LassoCV(alphas=αs, max_iter=5000))]),
    "Poly2_Ridge":   Pipeline([("prep", poly_preproc), ("ridge", RidgeCV(alphas=αs))]),
    "OLS_Selected":  Pipeline([("prep", preproc_base),
                               ("sel",  SelectFromModel(LassoCV(alphas=αs, max_iter=5000),
                                                        threshold="median")),
                               ("lr",   LinearRegression())]),
}

# —— 神经网络 ——
for n, cfg in [
        ("MLP_relu_64",     dict(hidden_layer_sizes=(64,),      activation="relu")),
        ("MLP_tanh_64_32",  dict(hidden_layer_sizes=(64, 32),   activation="tanh")),
        ("MLP_relu_128_64", dict(hidden_layer_sizes=(128, 64),  activation="relu"))]:
    models[n] = scaled(
        Pipeline([("prep", preproc_base),
                  ("mlp",  MLPRegressor(max_iter=1000, random_state=42, **cfg))]))

# —— 决策树 & 传统集成 ——
base_tree = DecisionTreeRegressor(random_state=42)
models.update({
    "CART":      scaled(Pipeline([("prep", preproc_base), ("tree", base_tree)])),
    "Bagging":   scaled(Pipeline([("prep", preproc_base),
                                  ("bag",  BaggingRegressor(estimator=base_tree,
                                                            n_estimators=50,
                                                            random_state=42))])),
    "AdaBoost":  scaled(Pipeline([("prep", preproc_base),
                                  ("ada",  AdaBoostRegressor(random_state=42))])),
})
for r in [0.3, 0.5, 0.7, 1.0]:
    models[f"RF_maxFeat_{r:.1f}D"] = scaled(
        Pipeline([("prep", preproc_base),
                  ("rf", RandomForestRegressor(n_estimators=200,
                                               max_features=r,
                                               n_jobs=-1,
                                               random_state=42))]))

# —— 梯度提升 / 极随机森林 ——
models["GBR"] = scaled(Pipeline([("prep", preproc_base),
                                 ("gbr",  GradientBoostingRegressor(random_state=42))]))

models["HistGB"] = scaled(Pipeline([("prep", preproc_base),
                                    ("hgb",  HistGradientBoostingRegressor(
                                                max_depth=None,
                                                learning_rate=0.05,
                                                l2_regularization=0.0,
                                                random_state=42))]))

models["ExtraTrees"] = scaled(Pipeline([("prep", preproc_base),
                                        ("etr",  ExtraTreesRegressor(
                                                    n_estimators=300,
                                                    min_samples_leaf=1,
                                                    n_jobs=-1,
                                                    random_state=42))]))

if XGBRegressor is not None:
    models["XGB"] = scaled(Pipeline([("prep", preproc_base),
                                     ("xgb",  XGBRegressor(
                                                  n_estimators=500,
                                                  learning_rate=0.05,
                                                  max_depth=6,
                                                  subsample=0.8,
                                                  colsample_bytree=0.8,
                                                  reg_lambda=1.0,
                                                  objective="reg:squarederror",
                                                  random_state=42,
                                                  n_jobs=-1))]))

if LGBMRegressor is not None:
    models["LightGBM"] = scaled(Pipeline([("prep", preproc_base),
                                          ("lgbm", LGBMRegressor(
                                                      n_estimators=500,
                                                      learning_rate=0.05,
                                                      max_depth=-1,
                                                      num_leaves=31,
                                                      subsample=0.8,
                                                      colsample_bytree=0.8,
                                                      random_state=42,
                                                      n_jobs=-1))]))

# 9. 评估与绘图
def evaluate_and_plot(name, pipe):
    """训练、评估、保存模型与图表（无交互式窗口）"""
    pipe.fit(X_tr, y_tr)
    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")

    y_pred_tr = pipe.predict(X_tr)
    y_pred_te = pipe.predict(X_te)

    def metrics(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        r2   = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    tr_MAE, tr_MSE, tr_RMSE, tr_R2 = metrics(y_tr, y_pred_tr)
    te_MAE, te_MSE, te_RMSE, te_R2 = metrics(y_te, y_pred_te)
    _, p_val = stats.ttest_1samp(y_te - y_pred_te, 0.0)

    pred_mean, pred_var = y_pred_te.mean(), y_pred_te.var()
    true_mean, true_var = y_te.mean(),   y_te.var()

    # —— 终端输出 —— 
    print(f"\n▶ {name}")
    print(f"   Test MAE {te_MAE:.2f} | MSE {te_MSE:.2f} | RMSE {te_RMSE:.2f} | R² {te_R2:.3f} | p {p_val:.3g}")
    print(f"   Mean pred {pred_mean:.2f} vs true {true_mean:.2f}")
    print(f"   Var  pred {pred_var :.2f} vs true {true_var :.2f}")

    # —— 图 —— 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_te, y_pred_te, alpha=.6)
    lims = [min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())]
    ax[0].plot(lims, lims, ' k', lw=1)
    ax[0].set_title("True vs Pred"); ax[0].set_xlabel("True"); ax[0].set_ylabel("Pred")
    sns.residplot(x=y_pred_te, y=y_te - y_pred_te, lowess=True,
                  ax=ax[1], color='#2ca02c', line_kws={'lw':1})
    ax[1].axhline(0, ls=' ', c='r'); ax[1].set_title("Residuals")
    fig.suptitle(name); fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = PLOTS_DIR / f"{name}_combined.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)         
    return dict(Model=name,
                Train_MAE=tr_MAE, Train_MSE=tr_MSE, Train_RMSE=tr_RMSE, Train_R2=tr_R2,
                Test_MAE=te_MAE,  Test_MSE=te_MSE,  Test_RMSE=te_RMSE,  Test_R2=te_R2,
                Pred_Mean=pred_mean, True_Mean=true_mean,
                Pred_Var=pred_var,   True_Var=true_var,
                Residual_p=p_val,
                png_path=png_path)

# 10. 训练循环
results = [evaluate_and_plot(n, m) for n, m in models.items()]

# 11. 汇总指标
summary = (pd.DataFrame(results)
           .sort_values("Test_RMSE")
           .reset_index(drop=True)
           .drop(columns="png_path"))
summary.to_csv("metrics_summary.csv", index=False)
print("\n===== 指标汇总 (按 Test_RMSE 升序) =====")
print(summary.to_string(index=False))

# 12. 总览：每页 6 张图
pngs   = [r["png_path"] for r in results]
chunks = [pngs[i:i+6] for i in range(0, len(pngs), 6)]
for idx, grp in enumerate(chunks, 1):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, p in zip(axes.flat, grp):
        img = plt.imread(p); ax.imshow(img); ax.set_axis_off()
        ax.set_title(Path(p).stem.replace("_combined", ""), fontsize=9)
    for ax in axes.flat[len(grp):]: ax.remove()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"all_models_part{idx}.png", dpi=150)
    plt.close(fig)

print("\nAll tasks completed. Images saved to 'plots/', models to 'models/', metrics to 'metrics_summary.csv'.")

import os, platform, webbrowser
for idx in range(1, len(chunks)+1):
    img_path = PLOTS_DIR / f"all_models_part{idx}.png"
    if platform.system() == "Windows":
        os.startfile(img_path)       
    else:
        webbrowser.open(img_path.as_uri()) 
