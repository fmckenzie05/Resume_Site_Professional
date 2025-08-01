# Machine Learning for Predictive Maintenance in Logistics: A Real-World Implementation

**Published:** August 14, 2021  
**Author:** Fernando McKenzie  
**Tags:** Machine Learning, Predictive Maintenance, IoT, TensorFlow, Supply Chain

## Introduction

After stabilizing our remote infrastructure in 2020, we turned our attention to leveraging machine learning for operational excellence. This article details our implementation of predictive maintenance systems for our logistics fleet and warehouse equipment, resulting in 35% reduction in unplanned downtime and $1.2M annual savings.

## The Maintenance Challenge

### Traditional Reactive Approach
- **Equipment failures** caused 2-4 hour delays per incident
- **Spare parts inventory** worth $800K sitting idle "just in case"
- **Preventive maintenance** performed on fixed schedules regardless of actual condition
- **Manual inspections** subjective and inconsistent

### Business Impact of Downtime
```
Annual Downtime Analysis (2020):
├── Forklift failures:        45 incidents × 3.2 hours = 144 hours
├── Conveyor belt issues:     23 incidents × 6.1 hours = 140 hours  
├── Loading dock problems:    31 incidents × 2.8 hours = 87 hours
├── HVAC system failures:     12 incidents × 8.5 hours = 102 hours
└── Total operational impact: 473 hours ($2.1M in lost productivity)
```

## Machine Learning Architecture

### Data Collection Infrastructure

**IoT Sensor Deployment:**
```python
# IoT sensor data collection system
import asyncio
import aiohttp
import json
from datetime import datetime
import boto3

class IoTSensorCollector:
    def __init__(self):
        self.kinesis_client = boto3.client('kinesis')
        self.sensor_endpoints = {
            'forklifts': ['192.168.1.10', '192.168.1.11', '192.168.1.12'],
            'conveyors': ['192.168.1.20', '192.168.1.21'],
            'hvac': ['192.168.1.30', '192.168.1.31'],
            'loading_docks': ['192.168.1.40', '192.168.1.41', '192.168.1.42']
        }
    
    async def collect_sensor_data(self, equipment_type, sensor_ip):
        """Collect data from individual sensor"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://{sensor_ip}/api/metrics') as response:
                    data = await response.json()
                    
                    # Standardize data format
                    sensor_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'equipment_type': equipment_type,
                        'sensor_id': sensor_ip,
                        'metrics': {
                            'temperature': data.get('temperature'),
                            'vibration': data.get('vibration'),
                            'oil_pressure': data.get('oil_pressure'),
                            'runtime_hours': data.get('runtime_hours'),
                            'error_codes': data.get('error_codes', []),
                            'fuel_level': data.get('fuel_level'),
                            'battery_voltage': data.get('battery_voltage')
                        }
                    }
                    
                    # Stream to Kinesis for real-time processing
                    await self.send_to_kinesis(sensor_data)
                    
                    return sensor_data
                    
        except Exception as e:
            print(f"Failed to collect from {sensor_ip}: {e}")
            return None
    
    async def send_to_kinesis(self, data):
        """Send sensor data to Kinesis stream"""
        self.kinesis_client.put_record(
            StreamName='equipment-telemetry',
            Data=json.dumps(data),
            PartitionKey=data['sensor_id']
        )
    
    async def collect_all_sensors(self):
        """Collect data from all sensors concurrently"""
        tasks = []
        
        for equipment_type, sensors in self.sensor_endpoints.items():
            for sensor_ip in sensors:
                task = asyncio.create_task(
                    self.collect_sensor_data(equipment_type, sensor_ip)
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log collection statistics
        successful = sum(1 for r in results if isinstance(r, dict))
        failed = len(results) - successful
        
        print(f"Data collection: {successful} successful, {failed} failed")
        
        return results

# Run collector every 30 seconds
if __name__ == "__main__":
    collector = IoTSensorCollector()
    
    async def main():
        while True:
            await collector.collect_all_sensors()
            await asyncio.sleep(30)
    
    asyncio.run(main())
```

### Feature Engineering Pipeline

**Data Preprocessing:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
    
    def create_time_based_features(self, df):
        """Create time-based features for maintenance prediction"""
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour_of_day'].between(8, 17).astype(int)
        
        # Operating time calculations
        df = df.sort_values(['sensor_id', 'timestamp'])
        df['time_since_last_maintenance'] = df.groupby('sensor_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['cumulative_runtime'] = df.groupby('sensor_id')['runtime_hours'].cumsum()
        
        return df
    
    def create_rolling_features(self, df, windows=[5, 15, 30]):
        """Create rolling window features for trend analysis"""
        
        numeric_columns = ['temperature', 'vibration', 'oil_pressure', 'fuel_level', 'battery_voltage']
        
        for window in windows:
            for col in numeric_columns:
                # Rolling statistics
                df[f'{col}_rolling_mean_{window}'] = df.groupby('sensor_id')[col].rolling(window).mean().reset_index(0, drop=True)
                df[f'{col}_rolling_std_{window}'] = df.groupby('sensor_id')[col].rolling(window).std().reset_index(0, drop=True)
                df[f'{col}_rolling_min_{window}'] = df.groupby('sensor_id')[col].rolling(window).min().reset_index(0, drop=True)
                df[f'{col}_rolling_max_{window}'] = df.groupby('sensor_id')[col].rolling(window).max().reset_index(0, drop=True)
                
                # Rate of change
                df[f'{col}_rate_of_change_{window}'] = df.groupby('sensor_id')[col].pct_change(periods=window)
        
        return df
    
    def create_anomaly_features(self, df):
        """Create features to detect anomalous patterns"""
        
        # Temperature anomalies
        df['temp_anomaly'] = (df['temperature'] > df['temperature'].quantile(0.95)).astype(int)
        
        # Vibration anomalies  
        df['vibration_anomaly'] = (df['vibration'] > df['vibration'].quantile(0.90)).astype(int)
        
        # Multiple simultaneous anomalies
        df['multiple_anomalies'] = (
            df['temp_anomaly'] + df['vibration_anomaly']
        ).clip(0, 1)
        
        # Consecutive error codes
        df['error_code_count'] = df['error_codes'].apply(len)
        df['has_critical_errors'] = df['error_codes'].apply(
            lambda x: any(code in ['E001', 'E002', 'E003'] for code in x)
        ).astype(int)
        
        return df
    
    def prepare_features(self, df, equipment_type):
        """Complete feature preparation pipeline"""
        
        # Create all feature types
        df = self.create_time_based_features(df)
        df = self.create_rolling_features(df)  
        df = self.create_anomaly_features(df)
        
        # Remove rows with insufficient data for rolling features
        df = df.dropna()
        
        # Select features for this equipment type
        feature_columns = [col for col in df.columns if col not in 
                          ['timestamp', 'sensor_id', 'equipment_type', 'failure_occurred']]
        
        X = df[feature_columns]
        
        # Scale features
        if equipment_type not in self.scalers:
            self.scalers[equipment_type] = RobustScaler()
            X_scaled = self.scalers[equipment_type].fit_transform(X)
        else:
            X_scaled = self.scalers[equipment_type].transform(X)
        
        # Feature selection for this equipment type
        if equipment_type not in self.feature_selectors and 'failure_occurred' in df.columns:
            y = df['failure_occurred']
            self.feature_selectors[equipment_type] = SelectKBest(f_regression, k=20)
            X_selected = self.feature_selectors[equipment_type].fit_transform(X_scaled, y)
        elif equipment_type in self.feature_selectors:
            X_selected = self.feature_selectors[equipment_type].transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return X_selected, feature_columns
```

### Machine Learning Models

**Multi-Model Ensemble Approach:**
```python
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class PredictiveMaintenanceModel:
    def __init__(self, equipment_type):
        self.equipment_type = equipment_type
        self.models = {}
        self.ensemble_weights = {}
        
    def create_lstm_model(self, sequence_length, n_features):
        """Create LSTM model for time series prediction"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_sequence_data(self, X, y, sequence_length=48):
        """Create sequences for LSTM training"""
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train ensemble of models"""
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # SVM
        svm_model = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        self.models['svm'] = svm_model
        
        # LSTM (requires sequence data)
        sequence_length = 48  # 24 hours of 30-minute intervals
        X_train_seq, y_train_seq = self.create_sequence_data(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self.create_sequence_data(X_val, y_val, sequence_length)
        
        lstm_model = self.create_lstm_model(sequence_length, X_train.shape[1])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        self.models['lstm'] = lstm_model
        
        # Calculate ensemble weights based on validation performance
        self.calculate_ensemble_weights(X_val, y_val)
        
        return self.models
    
    def calculate_ensemble_weights(self, X_val, y_val):
        """Calculate weights for ensemble based on validation performance"""
        
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                # For LSTM, need sequence data
                sequence_length = 48
                X_val_seq, y_val_seq = self.create_sequence_data(X_val, y_val, sequence_length)
                predictions = model.predict(X_val_seq)
                score = roc_auc_score(y_val_seq, predictions)
            else:
                predictions = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, predictions)
            
            model_scores[name] = score
        
        # Convert scores to weights (higher score = higher weight)
        total_score = sum(model_scores.values())
        self.ensemble_weights = {
            name: score / total_score 
            for name, score in model_scores.items()
        }
        
        print(f"Ensemble weights for {self.equipment_type}:")
        for name, weight in self.ensemble_weights.items():
            print(f"  {name}: {weight:.3f}")
    
    def predict_failure_probability(self, X):
        """Make ensemble prediction"""
        
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                # For LSTM, need sequence data
                sequence_length = 48
                if len(X) >= sequence_length:
                    X_seq = X[-sequence_length:].reshape(1, sequence_length, -1)
                    pred = model.predict(X_seq)[0][0]
                else:
                    pred = 0.0  # Not enough data for LSTM prediction
            else:
                pred = model.predict_proba(X.reshape(1, -1))[0][1]
            
            predictions[name] = pred
        
        # Weighted ensemble prediction
        ensemble_pred = sum(
            predictions[name] * self.ensemble_weights[name]
            for name in predictions.keys()
        )
        
        return ensemble_pred, predictions
    
    def save_models(self, filepath):
        """Save trained models"""
        
        model_data = {
            'equipment_type': self.equipment_type,
            'ensemble_weights': self.ensemble_weights,
            'traditional_models': {
                name: model for name, model in self.models.items() 
                if name != 'lstm'
            }
        }
        
        # Save traditional models
        joblib.dump(model_data, f"{filepath}_{self.equipment_type}_models.pkl")
        
        # Save LSTM model separately
        if 'lstm' in self.models:
            self.models['lstm'].save(f"{filepath}_{self.equipment_type}_lstm.h5")
```

## Real-Time Monitoring System

### Alert and Response System:
```python
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk

class MaintenanceAlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            'critical': 0.85,  # 85% failure probability
            'warning': 0.65,   # 65% failure probability
            'watch': 0.45      # 45% failure probability
        }
        self.slack_client = slack_sdk.WebClient(token="your-slack-token")
        
    async def process_predictions(self, predictions):
        """Process model predictions and trigger alerts"""
        
        for equipment_id, prediction_data in predictions.items():
            failure_prob = prediction_data['failure_probability']
            equipment_type = prediction_data['equipment_type']
            
            alert_level = self.classify_alert_level(failure_prob)
            
            if alert_level:
                await self.send_alert(
                    equipment_id, 
                    equipment_type, 
                    failure_prob, 
                    alert_level,
                    prediction_data.get('contributing_factors', [])
                )
    
    def classify_alert_level(self, failure_prob):
        """Classify alert level based on failure probability"""
        
        if failure_prob >= self.alert_thresholds['critical']:
            return 'critical'
        elif failure_prob >= self.alert_thresholds['warning']:
            return 'warning'
        elif failure_prob >= self.alert_thresholds['watch']:
            return 'watch'
        else:
            return None
    
    async def send_alert(self, equipment_id, equipment_type, failure_prob, alert_level, factors):
        """Send maintenance alert through multiple channels"""
        
        message = self.create_alert_message(
            equipment_id, equipment_type, failure_prob, alert_level, factors
        )
        
        # Send Slack notification
        await self.send_slack_alert(message, alert_level)
        
        # Send email for critical alerts
        if alert_level == 'critical':
            await self.send_email_alert(message, equipment_id)
        
        # Create maintenance work order
        if alert_level in ['critical', 'warning']:
            await self.create_work_order(equipment_id, equipment_type, failure_prob, factors)
    
    def create_alert_message(self, equipment_id, equipment_type, failure_prob, alert_level, factors):
        """Create formatted alert message"""
        
        emoji_map = {
            'critical': '🚨',
            'warning': '⚠️',
            'watch': '👀'
        }
        
        message = f"""
{emoji_map[alert_level]} **{alert_level.upper()} MAINTENANCE ALERT**

**Equipment:** {equipment_type} (ID: {equipment_id})
**Failure Probability:** {failure_prob:.1%}
**Alert Level:** {alert_level.title()}

**Contributing Factors:**
{chr(10).join(f"• {factor}" for factor in factors)}

**Recommended Actions:**
{self.get_recommended_actions(alert_level, equipment_type)}

**View Dashboard:** http://maintenance-dashboard.internal/equipment/{equipment_id}
        """
        
        return message
    
    def get_recommended_actions(self, alert_level, equipment_type):
        """Get recommended actions based on alert level and equipment type"""
        
        actions = {
            'critical': {
                'forklift': [
                    "Schedule immediate inspection",
                    "Check hydraulic fluid levels",
                    "Inspect battery connections",
                    "Review operator logs"
                ],
                'conveyor': [
                    "Stop conveyor for inspection",
                    "Check belt tension and alignment",
                    "Inspect motor bearings",
                    "Test emergency stops"
                ]
            },
            'warning': {
                'forklift': [
                    "Schedule inspection within 24 hours",
                    "Monitor vibration levels",
                    "Check tire pressure",
                    "Review usage patterns"
                ],
                'conveyor': [
                    "Schedule inspection within 48 hours",
                    "Monitor belt speed variations",
                    "Check lubrication levels",
                    "Test safety sensors"
                ]
            }
        }
        
        equipment_actions = actions.get(alert_level, {}).get(equipment_type, [
            f"Schedule {alert_level} level inspection",
            "Review recent maintenance history",
            "Monitor closely for changes"
        ])
        
        return chr(10).join(f"• {action}" for action in equipment_actions)
    
    async def send_slack_alert(self, message, alert_level):
        """Send alert to Slack channel"""
        
        channel_map = {
            'critical': '#maintenance-critical',
            'warning': '#maintenance-alerts', 
            'watch': '#maintenance-watch'
        }
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel_map[alert_level],
                text=message,
                username="MaintenanceBot"
            )
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    async def create_work_order(self, equipment_id, equipment_type, failure_prob, factors):
        """Create maintenance work order in CMMS system"""
        
        work_order = {
            'equipment_id': equipment_id,
            'equipment_type': equipment_type,
            'priority': 'high' if failure_prob > 0.75 else 'medium',
            'description': f"Predictive maintenance - {failure_prob:.1%} failure probability",
            'contributing_factors': factors,
            'estimated_hours': self.estimate_maintenance_hours(equipment_type, factors),
            'required_parts': self.suggest_parts(equipment_type, factors),
            'created_by': 'ML_Prediction_System',
            'due_date': self.calculate_due_date(failure_prob)
        }
        
        # Submit to CMMS API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://cmms.internal/api/work-orders',
                json=work_order,
                headers={'Authorization': 'Bearer your-token'}
            ) as response:
                if response.status == 201:
                    print(f"Work order created for {equipment_id}")
                else:
                    print(f"Failed to create work order: {response.status}")
```

## Results and Business Impact

### Performance Metrics (12 months post-implementation)

**Prediction Accuracy:**
```
Model Performance by Equipment Type:
├── Forklifts:      87% accuracy, 0.91 AUC-ROC
├── Conveyors:      92% accuracy, 0.94 AUC-ROC  
├── HVAC Systems:   85% accuracy, 0.89 AUC-ROC
└── Loading Docks:  83% accuracy, 0.87 AUC-ROC

Overall System Performance:
├── True Positives:  156 failures correctly predicted
├── False Positives: 23 unnecessary maintenance events
├── False Negatives: 18 failures missed
├── True Negatives:  1,247 correctly identified as healthy
```

**Operational Improvements:**
```python
# Before vs After Analysis
maintenance_metrics = {
    'unplanned_downtime': {
        'before': 473,  # hours per year
        'after': 307,   # hours per year  
        'improvement': '35% reduction'
    },
    'maintenance_costs': {
        'before': 2100000,  # $2.1M per year
        'after': 1650000,   # $1.65M per year
        'savings': 450000   # $450K annual savings
    },
    'spare_parts_inventory': {
        'before': 800000,   # $800K inventory
        'after': 550000,    # $550K inventory
        'reduction': 250000 # $250K freed up capital
    },
    'maintenance_efficiency': {
        'before': 0.72,     # 72% planned maintenance
        'after': 0.89,      # 89% planned maintenance
        'improvement': '24% increase in planned work'
    }
}
```

### Cost-Benefit Analysis

**Implementation Costs:**
```
Year 1 Investment:
├── IoT sensors and installation:    $125,000
├── ML infrastructure (AWS):         $48,000
├── Software development:            $180,000
├── Training and change management:  $35,000
├── Integration costs:               $25,000
└── Total investment:                $413,000

Annual Operating Costs:
├── Cloud infrastructure:            $62,000
├── Sensor maintenance:              $15,000
├── Software licenses:               $28,000
└── Total annual:                    $105,000
```

**Financial Returns:**
```
Annual Benefits:
├── Reduced unplanned downtime:      $750,000
├── Lower maintenance costs:         $450,000
├── Freed inventory capital:         $250,000
├── Improved efficiency:             $180,000
└── Total annual benefits:           $1,630,000

ROI Calculation:
├── Year 1 ROI: 295% ((1,630,000 - 413,000) / 413,000)
├── Payback period: 4.9 months
└── 3-year NPV: $4.2M (assuming 8% discount rate)
```

## Lessons Learned and Best Practices

### 1. Data Quality is Critical
**Challenge:** Initial sensor data had 15% noise/error rate

**Solution:**
```python
# Data quality monitoring
class DataQualityMonitor:
    def __init__(self):
        self.quality_thresholds = {
            'missing_data_rate': 0.05,  # Max 5% missing
            'outlier_rate': 0.02,       # Max 2% outliers
            'sensor_drift': 0.1         # Max 10% drift from baseline
        }
    
    def validate_sensor_data(self, sensor_data):
        """Validate incoming sensor data quality"""
        
        quality_score = 1.0
        issues = []
        
        # Check for missing values
        missing_rate = sensor_data.isnull().sum() / len(sensor_data)
        if missing_rate > self.quality_thresholds['missing_data_rate']:
            quality_score -= 0.3
            issues.append(f"High missing data rate: {missing_rate:.2%}")
        
        # Check for outliers using IQR method
        numeric_columns = sensor_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = sensor_data[col].quantile(0.25)
            Q3 = sensor_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = sensor_data[
                (sensor_data[col] < Q1 - 1.5 * IQR) | 
                (sensor_data[col] > Q3 + 1.5 * IQR)
            ]
            outlier_rate = len(outliers) / len(sensor_data)
            
            if outlier_rate > self.quality_thresholds['outlier_rate']:
                quality_score -= 0.2
                issues.append(f"High outlier rate in {col}: {outlier_rate:.2%}")
        
        return quality_score, issues
```

### 2. Human-in-the-Loop Validation
**Learning:** Maintenance technicians' domain expertise improved model accuracy by 12%

**Implementation:**
- Weekly model review sessions with maintenance team
- Feedback loop for false positives/negatives
- Technician can override predictions with justification

### 3. Gradual Rollout Strategy
**Approach:** Started with non-critical equipment, expanded based on confidence

**Timeline:**
- Month 1-2: HVAC systems (lowest risk)
- Month 3-4: Conveyors (medium risk)  
- Month 5-6: Forklifts (highest impact)
- Month 7+: Full fleet deployment

## Future Enhancements (2022 Roadmap)

### 1. Advanced Analytics
```python
# Failure root cause analysis
def analyze_failure_patterns():
    """Use ML to identify common failure patterns and root causes"""
    
    failure_data = get_historical_failures()
    
    # Cluster analysis to find failure patterns
    from sklearn.cluster import KMeans
    
    features = ['temperature_pattern', 'vibration_signature', 'usage_intensity']
    kmeans = KMeans(n_clusters=5)
    failure_clusters = kmeans.fit_predict(failure_data[features])
    
    # Associate clusters with maintenance actions
    cluster_actions = {
        0: "Lubrication system maintenance",
        1: "Belt/chain replacement",
        2: "Motor bearing replacement", 
        3: "Hydraulic system service",
        4: "Electrical component check"
    }
    
    return cluster_actions
```

### 2. Supply Chain Integration
```python
# Predictive parts ordering
def predict_parts_demand():
    """Predict spare parts demand based on failure predictions"""
    
    failure_predictions = get_all_equipment_predictions()
    
    parts_demand = {}
    for equipment_id, prediction in failure_predictions.items():
        if prediction['failure_probability'] > 0.6:
            required_parts = get_parts_for_equipment(equipment_id)
            for part in required_parts:
                parts_demand[part] = parts_demand.get(part, 0) + 1
    
    # Trigger automatic procurement for high-demand parts
    for part, demand in parts_demand.items():
        current_stock = get_current_stock(part)
        if demand > current_stock * 0.5:  # If demand > 50% of stock
            create_purchase_order(part, demand)
```

### 3. Mobile Integration
- Real-time alerts on maintenance team mobile devices
- Augmented reality overlays for equipment diagnostics
- Voice-activated work order updates

## Conclusion

Our predictive maintenance implementation transformed reactive firefighting into proactive optimization. The combination of IoT sensors, machine learning, and human expertise created a system that not only reduces costs but enhances operational reliability.

**Key Success Factors:**
- **Start with high-value, low-risk equipment** to build confidence
- **Invest in data quality** from day one
- **Include domain experts** in model development
- **Focus on actionable insights** over algorithmic complexity
- **Measure business impact** continuously

The $1.2M annual savings and 35% reduction in unplanned downtime demonstrated clear ROI, but the strategic value goes beyond cost savings. We now have a foundation for more advanced analytics and are positioned to expand predictive capabilities across our entire operation.

**2021 taught us:** Machine learning works best when it augments human expertise rather than replacing it. The most successful predictions came from models that learned from both sensor data and technician insights.

---

*Implementing predictive maintenance? Let's connect on [LinkedIn](https://www.linkedin.com/in/fernandomckenzie/) to discuss your ML strategy.* 