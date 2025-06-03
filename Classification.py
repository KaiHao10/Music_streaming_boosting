import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from category_encoders import CatBoostEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import shap

# 加载数据
file=r"D:\Desktop\Study\MachineLearning\Project\data.csv"
data = pd.read_csv(file)

data.columns = [
    'User_ID', 'Age', 'Country', 'Streaming_Platform', 'Top_Genre', 
    'Minutes_Streamed', 'Songs_Liked', 'Most_Played_Artist', 
    'Subscription_Type', 'Listening_Time', 'Discover_Engagement', 
    'Repeat_Rate'
]

data['Active_Rate'] = data['Minutes_Streamed'] / (data['Songs_Liked'] + 1e-5)
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 100], labels=['Teen', 'Young', 'Adult', 'Senior'])
data['Streaming_Intensity'] = data['Minutes_Streamed'] / (data['Age'] + 1)  # 考虑年龄因素
data['Engagement_Score'] = data['Discover_Engagement'] * data['Repeat_Rate']
artist_counts = data['Most_Played_Artist'].value_counts()
data['Artist_Popularity'] = data['Most_Played_Artist'].map(artist_counts)


# 处理分类变量（训练集独立处理）
X = data.drop(['Subscription_Type', 'User_ID'], axis=1)
y = data['Subscription_Type']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 对高基数特征归并低频类别
high_card_cols = ['Most_Played_Artist', 'Top_Genre']
for col in high_card_cols:
    # 使用训练集计算阈值
    freq = X_train[col].value_counts(normalize=True)
    cum_freq = freq.cumsum()
    threshold = cum_freq[cum_freq <= 0.8].index
    
    # 应用替换
    X_train.loc[:, col] = X_train[col].where(X_train[col].isin(threshold), "Other")
    X_test.loc[:, col] = X_test[col].where(X_test[col].isin(threshold), "Other")


# 定义预处理器
numeric_features = ['Age', 'Minutes_Streamed', 'Songs_Liked', 
                   'Discover_Engagement', 'Repeat_Rate', 'Active_Rate','Streaming_Intensity', 'Engagement_Score', 'Artist_Popularity']
categorical_low = ['Country', 'Streaming_Platform','Age_Group']
categorical_high = ['Most_Played_Artist', 'Top_Genre']
categorical_ordinal = ['Listening_Time'] 
ordinal_order = [['Morning', 'Afternoon', 'Night']] 
cluster_features = [
    'Minutes_Streamed', 'Songs_Liked', 'Discover_Engagement',
    'Repeat_Rate', 'Active_Rate', 'Streaming_Intensity',
    'Engagement_Score', 'Artist_Popularity'
]

# 初始化聚类器（使用训练集拟合）
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train[cluster_features])

# 为训练集添加聚类特征
train_cluster_labels = kmeans.predict(X_train[cluster_features])
train_cluster_dist = kmeans.transform(X_train[cluster_features])
X_train['Cluster_Label'] = train_cluster_labels
for i in range(kmeans.n_clusters):
    X_train[f'Cluster_Dist_{i}'] = train_cluster_dist[:, i]

# 为测试集添加聚类特征（使用训练好的模型）
test_cluster_labels = kmeans.predict(X_test[cluster_features])
test_cluster_dist = kmeans.transform(X_test[cluster_features])
X_test['Cluster_Label'] = test_cluster_labels
for i in range(kmeans.n_clusters):
    X_test[f'Cluster_Dist_{i}'] = test_cluster_dist[:, i]


# 在现有特征列表中添加新聚类特征
numeric_features += ['Cluster_Label'] + [f'Cluster_Dist_{i}' for i in range(5)]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(output_distribution='normal'))
        ]),  
        ['Age', 'Minutes_Streamed', 'Songs_Liked', 
         'Discover_Engagement', 'Repeat_Rate', 'Active_Rate'] 
        ),
        ('cat_low', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),  
        ['Country', 'Streaming_Platform', 'Listening_Time', 'Age_Group'] 
        ),
        ('cat_high', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CatBoostEncoder(sigma=0.1))
        ]), ['Most_Played_Artist', 'Top_Genre']
        ),
        ('cat_ordinal', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=ordinal_order))
        ]), ['Listening_Time'])
    ],
    sparse_threshold=0 
)

cluster_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=5)),
    ('classifier', LGBMClassifier())
])

# 定义模型和参数网格
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'params': {
            'classifier__C': np.logspace(-4, 4, 9),
            'feature_selection__k': [10, 'all']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'feature_selection__k': [15, 'all']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(
            scale_pos_weight=(sum(y == 0) / sum(y == 1)),
            eval_metric='logloss'
        ),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'feature_selection__k': [15, 'all']
        }
    },
    'SVM': {
        'model': SVC(
            class_weight='balanced',
            probability=True, 
            random_state=42
        ),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'feature_selection__k': [15, 'all']
        }
    },
    # 新增LightGBM模型
    'LightGBM': {
        'model': LGBMClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  
        ),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__num_leaves': [31, 63],
            'classifier__max_depth': [5, 10],
            'feature_selection__k': [15, 'all']
        }
    }
}

# 训练和评估
results = {}
best_models = {}

for model_name, config in models.items():
    pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', RandomOverSampler(
        random_state=42,
        sampling_strategy='not majority'
    )),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', config['model'])
])
    
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        error_score='raise'
    )

    try:
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        
        # 评估
        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)[:, 1]
        
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'report': classification_report(y_test, y_pred, target_names=le.classes_)
        }
    except Exception as e:
        print(f"训练 {model_name} 失败: {str(e)}")
        continue

# 输出结果
for model, res in results.items():
    print(f"\n{'-'*40}\nModel: {model}")
    print(f"Accuracy: {res['accuracy']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")
    print(res['report'])

fitted_preprocessor = best_models['LightGBM'].named_steps['preprocessor']

# 使用已拟合的预处理器转换训练数据
X_processed = fitted_preprocessor.transform(X_train)  # 注意：这里使用训练集数据

# 获取正确的特征名称
feature_names = fitted_preprocessor.get_feature_names_out()

# 创建SHAP解释器（使用LightGBM模型）
explainer = shap.TreeExplainer(best_models['LightGBM'].named_steps['classifier'])  
shap_values = explainer.shap_values(X_processed)  

# 绘制SHAP摘要图  
shap.summary_plot(  
    shap_values,  
    X_processed,  
    feature_names=feature_names,  # 使用正确的特征名称
    plot_type="dot",  
    max_display=15,  
    show=False  
)  
plt.title("High Feature Importance ≠ Predictive Power (AUC=0.49)", fontsize=12)  
plt.savefig('shap_paradox.png', dpi=300, bbox_inches='tight')  
plt.show()

# ====== 后续分析同样使用已拟合的预处理器 ======
# 计算所有模型的特征重要性  
paradox_data = []  
tree_models = ['Random Forest', 'LightGBM', 'XGBoost']  # 只处理这些树模型

for model_name in tree_models:  
    # 获取当前模型的已拟合预处理器
    model_preprocessor = best_models[model_name].named_steps['preprocessor']
    X_processed_all = model_preprocessor.transform(X_train)
    
    # 获取特征重要性
    model = best_models[model_name].named_steps['classifier']  
    
    # 获取特征名称
    feat_names = model_preprocessor.get_feature_names_out()
    
    # 获取特征重要性值（不同模型可能有不同的属性名）
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 对于线性模型，使用系数的绝对值作为重要性
        importance = np.abs(model.coef_[0])
    else:
        # 跳过不支持特征重要性的模型
        continue
    
    roc_auc = results[model_name]['roc_auc']  
    
    for i, feat in enumerate(feat_names):  # 遍历所有特征
        paradox_data.append({  
            'Feature': feat,  
            'Importance': importance[i] if i < len(importance) else 0,  
            'Model': model_name,  
            'AUC': roc_auc  
        })  
        
df_paradox = pd.DataFrame(paradox_data)  

# 提取聚类特征  
cluster_feats = [f for f in df_paradox['Feature'].unique() if 'Cluster' in f]  
cluster_importance = []  

# 只处理树模型的特征重要性
for model_name in tree_models:  
    if model_name in best_models:
        model = best_models[model_name].named_steps['classifier']  
        
        # 获取特征名称
        feat_names = best_models[model_name].named_steps['preprocessor'].get_feature_names_out()
        
        # 获取特征重要性值
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        else:
            continue
        
        # 找到聚类特征的索引
        idx = [i for i, f in enumerate(feat_names) if f in cluster_feats]
        
        if idx:  # 如果有聚类特征
            cluster_importance.append(np.sum(imp[idx]))  
        else:
            cluster_importance.append(0)

# 绘制热力图  
plt.figure(figsize=(8, 4))  
sns.heatmap(  
    pd.DataFrame({  
        'Model': tree_models,  
        'Cluster_Contribution': cluster_importance,  
        'AUC': [results[m]['roc_auc'] for m in tree_models if m in results]  
    }).set_index('Model'),  
    annot=True,  
    cmap='YlOrRd',  
    fmt='.2f'  
)  
plt.title("Cluster Features Contribution", pad=20)  
plt.savefig('cluster_heatmap.png', dpi=300)
plt.show()