import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SAMPLE_SIZE = 100000

print(f"Loading US Accidents dataset (first {SAMPLE_SIZE:,} rows for faster training)...")
print("For production use, remove the nrows parameter to train on full dataset")
df = pd.read_csv('traffic_accidents.csv', nrows=SAMPLE_SIZE)
initial_shape = df.shape
print(f"Initial shape: {initial_shape}")

df = df.dropna(subset=['Severity', 'Start_Time', 'Weather_Condition'])
df = df.drop_duplicates()
print(f"After cleaning: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")

severity_map = {1: 0, 2: 1, 3: 2, 4: 3}
df['severity'] = df['Severity'].map(severity_map)
df = df.dropna(subset=['severity'])
df['severity'] = df['severity'].astype(int)
print(f"\nSeverity distribution:")
print(df['severity'].value_counts().sort_index())

df['crash_date'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df = df.dropna(subset=['crash_date'])
df['Hour'] = df['crash_date'].dt.hour
df['Day_of_Week'] = df['crash_date'].dt.dayofweek + 1
df['Month'] = df['crash_date'].dt.month
df['Year'] = df['crash_date'].dt.year
df['Is_Weekend'] = (df['Day_of_Week'].isin([6,7])).astype(int)
df['Is_Rush_Hour'] = (df['Hour'].isin([7,8,9,16,17,18])).astype(int)
df['Season'] = df['Month'].apply(lambda x: 1 if x in [12,1,2] else (2 if x in [3,4,5] else (3 if x in [6,7,8] else 4)))

print(f"\nTime features created successfully")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar([0,1,2,3], df['severity'].value_counts().sort_index(), color=['green','yellow','orange','red'])
plt.title('Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Accidents')
plt.subplot(1,2,2)
plt.hist(df['Hour'], bins=24, color='blue', edgecolor='black')
plt.title('Hourly Distribution')
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('models/severity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Severity plots saved")

df['weather_condition'] = df['Weather_Condition'].fillna('Clear')
df['lighting_condition'] = df['Sunrise_Sunset'].fillna('Day')
df['traffic_control_device'] = df.apply(lambda x: 'Traffic_Signal' if x['Traffic_Signal'] else ('Stop_Sign' if x['Stop'] else 'No_Control'), axis=1)
df['roadway_surface_cond'] = df.apply(lambda x: 'Wet' if 'Rain' in str(x['Weather_Condition']) else ('Snow' if 'Snow' in str(x['Weather_Condition']) else 'Dry'), axis=1)

df['num_units'] = 1
df['injuries_total'] = df['Severity']
df['injuries_fatal'] = (df['Severity'] == 4).astype(int)
df['injuries_incapacitating'] = (df['Severity'] == 3).astype(int)
df['injuries_non_incapacitating'] = (df['Severity'] == 2).astype(int)
df['injuries_reported_not_evident'] = 0
df['injuries_no_indication'] = (df['Severity'] == 1).astype(int)
df['first_crash_type'] = 'Unknown'
df['trafficway_type'] = 'Standard'
df['alignment'] = 'Straight'
df['road_defect'] = 'No_Defect'
df['crash_type'] = 'Accident'
df['damage'] = df['Severity'].apply(lambda x: 'Major' if x >= 3 else 'Minor')
df['prim_contributory_cause'] = 'Weather' if df['Weather_Condition'].notna().any() else 'Unknown'
df['intersection_related_i'] = df['Junction'].astype(int)

print(f"\nFeature engineering completed")

cols = ['weather_condition','lighting_condition','traffic_control_device','first_crash_type',
        'trafficway_type','alignment','roadway_surface_cond','road_defect','crash_type',
        'damage','prim_contributory_cause','intersection_related_i']

le = {}
for c in cols:
    le[c] = LabelEncoder()
    df[c] = le[c].fit_transform(df[c].astype(str))

print(f"Label encoding completed for {len(le)} categorical features")

feats = ['num_units','injuries_total','injuries_fatal','injuries_incapacitating',
         'injuries_non_incapacitating','injuries_reported_not_evident','injuries_no_indication',
         'Hour','Day_of_Week','Month','Year','Is_Weekend','Is_Rush_Hour','Season',
         'weather_condition','lighting_condition','traffic_control_device','first_crash_type',
         'trafficway_type','alignment','roadway_surface_cond','road_defect','crash_type',
         'damage','prim_contributory_cause','intersection_related_i']

X = df[feats]
y = df['severity']

nums = ['num_units','injuries_total','injuries_fatal','injuries_incapacitating',
        'injuries_non_incapacitating','injuries_reported_not_evident','injuries_no_indication',
        'Hour','Day_of_Week','Month','Year','Is_Weekend','Is_Rush_Hour']

sc = StandardScaler()
X[nums] = sc.fit_transform(X[nums])

print(f"\nFeature scaling completed")
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
sm = SMOTE(random_state=42)
X_train,y_train = sm.fit_resample(X_train,y_train)
print(f"\nTrain/test split completed")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Train classes: {sorted(y_train.unique())}")
print(f"Test classes: {sorted(y_test.unique())}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
rf.fit(X_train,y_train)
rf_acc = rf.score(X_test,y_test)
print(f'Random Forest Accuracy: {rf_acc:.4f}')

print("\nTraining XGBoost...")
xg = XGBClassifier(n_estimators=100,random_state=42,n_jobs=-1,eval_metric='mlogloss')
xg.fit(X_train,y_train)
xg_acc = xg.score(X_test,y_test)
print(f'XGBoost Accuracy: {xg_acc:.4f}')

print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100,random_state=42)
gb.fit(X_train,y_train)
gb_acc = gb.score(X_test,y_test)
print(f'Gradient Boosting Accuracy: {gb_acc:.4f}')

print(f'\n=== Model Comparison ===')
print(f'RF: {rf_acc:.4f}, XGB: {xg_acc:.4f}, GB: {gb_acc:.4f}')
plt.figure(figsize=(8,5))
plt.bar(['Random Forest','XGBoost','Gradient Boosting'], [rf_acc,xg_acc,gb_acc], color='teal')
plt.ylim(0.5,1.0)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Model comparison plot saved")

best = max([(rf_acc,'RF',rf),(xg_acc,'XGB',xg),(gb_acc,'GB',gb)])
print(f'\n=== Best Model ===')
print(f'Model: {best[1]}')
print(f'Accuracy: {best[0]:.4f}')

print("\nSaving models...")
joblib.dump(best[2],'models/accident_severity_model.pkl')
joblib.dump(sc,'models/feature_scaler.pkl')
joblib.dump(le,'models/label_encoders.pkl')
joblib.dump(feats,'models/feature_names.pkl')
joblib.dump({'model_type':best[1],'accuracy':best[0]},'models/model_metadata.pkl')
print('All models saved successfully!')

top_n = 12
importance = best[2].feature_importances_
top_idx = np.argsort(importance)[-top_n:]
plt.figure(figsize=(8,6))
plt.barh(range(top_n), importance[top_idx])
plt.yticks(range(top_n), [feats[i] for i in top_idx], fontsize=9)
plt.xlabel('Feature Importance')
plt.title('Top Feature Importances')
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved")

y_pred = best[2].predict(X_test)
print('\n=== Model Evaluation ===')
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Injury','Non-Incap','Incap','Fatal']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Injury','Minor','Serious','Fatal'],
            yticklabels=['No Injury','Minor','Serious','Fatal'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix plot saved")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Best Model: {best[1]}")
print(f"Accuracy: {best[0]:.4f}")
print(f"Models saved in: models/")
print("You can now run: streamlit run app.py")
