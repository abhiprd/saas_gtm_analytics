"""
Predictive LTV Model

This script builds a machine learning model to predict customer lifetime value (LTV)
using only information available at the time of customer acquisition (Day 0).

BUSINESS VALUE:
- Predict LTV at signup instead of waiting 17 months
- Set appropriate CAC limits by customer segment
- Prioritize high-value customers for sales/CS attention
- Optimize marketing spend toward high-LTV channels

MODEL APPROACH:
1. Prepare training data with Day-0 features and actual LTV outcomes
2. Engineer features from categorical variables (one-hot encoding)
3. Split data into train/test sets
4. Train multiple models and compare performance
5. Evaluate model accuracy and feature importance
6. Generate predictions for new customers
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def add_src_to_path():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    for _ in range(5):
        src_path = os.path.join(current_dir, 'src')
        if os.path.isdir(src_path):
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            return src_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    raise FileNotFoundError("Could not find 'src' directory.")

add_src_to_path()
from config import SYNTHETIC_DATA_PATH, PROCESSED_DATA_PATH

def prepare_training_data():
    """
    STEP 1: Prepare Training Data
    
    Load customer data and calculate actual LTV (target variable).
    Merge with Day-0 features that would be available at signup.
    """
    print("="*80)
    print("PREDICTIVE LTV MODEL - Training and Evaluation")
    print("="*80)
    
    print("\n[Step 1] Loading data...")
    
    # Load raw data
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
    
    # Calculate actual LTV (sum of all MRR for each customer)
    actual_ltv = revenue_df.groupby('account_id').agg({
        'MRR': 'sum',
        'tenure_months': 'max'
    }).reset_index()
    actual_ltv.columns = ['account_id', 'actual_ltv', 'actual_tenure']
    
    # Merge with Day-0 features
    modeling_data = accounts_df.merge(actual_ltv, on='account_id')
    
    print(f"   ✓ Loaded {len(modeling_data):,} customers")
    print(f"   ✓ Average LTV: ${modeling_data['actual_ltv'].mean():,.2f}")
    
    return modeling_data

def engineer_features(df):
    """
    STEP 2: Feature Engineering
    
    Transform raw features into model inputs:
    - One-hot encode categorical variables (channel, plan)
    - Keep numeric features as-is (quality score, seats)
    - Create interaction features if needed
    """
    print("\n[Step 2] Engineering features...")
    
    # Select features available at Day 0
    features_df = df[[
        'acquisition_channel',
        'initial_plan',
        'initial_seats',
        'latent_quality_score'
    ]].copy()
    
    # One-hot encode categorical variables
    features_encoded = pd.get_dummies(
        features_df, 
        columns=['acquisition_channel', 'initial_plan'],
        drop_first=False  # Keep all categories for interpretability
    )
    
    # Target variable
    y = df['actual_ltv']
    
    print(f"   ✓ Created {features_encoded.shape[1]} features")
    print(f"   ✓ Feature names: {features_encoded.columns.tolist()}")
    
    return features_encoded, y, features_encoded.columns.tolist()

def train_models(X_train, X_test, y_train, y_test):
    """
    STEP 3: Train Multiple Models
    
    Train three different algorithms and compare performance:
    1. Linear Regression - Simple baseline
    2. Random Forest - Handles non-linear relationships
    3. Gradient Boosting - Often best performance
    """
    print("\n[Step 3] Training models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'predictions': y_pred_test
        }
        
        print(f"      Train R²: {train_r2:.3f}")
        print(f"      Test R²: {test_r2:.3f}")
        print(f"      MAE: ${test_mae:,.0f}")
    
    return results

def evaluate_best_model(results, X_test, y_test, feature_names):
    """
    STEP 4: Evaluate Best Model
    
    Select the best performing model and analyze:
    - Prediction accuracy
    - Feature importance
    - Error distribution
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Find best model by test R²
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_model_name]
    best_model = best_result['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {best_result['test_r2']:.3f}")
    print(f"   Mean Absolute Error: ${best_result['test_mae']:,.0f}")
    print(f"   Root Mean Squared Error: ${best_result['test_rmse']:,.0f}")
    
    # Interpretation of R²
    r2 = best_result['test_r2']
    if r2 > 0.8:
        interpretation = "EXCELLENT - Model explains >80% of LTV variation"
    elif r2 > 0.6:
        interpretation = "GOOD - Model is useful for decision-making"
    elif r2 > 0.4:
        interpretation = "FAIR - Model provides some signal but has limitations"
    else:
        interpretation = "POOR - Model needs improvement"
    
    print(f"\n   Interpretation: {interpretation}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 Top 10 Most Important Features:")
        
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']:40s} {row['importance']:.3f}")
    
    # Prediction accuracy by LTV bucket
    print("\n📈 Prediction Accuracy by LTV Range:")
    
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': best_result['predictions']
    })
    
    # Create LTV buckets
    predictions_df['ltv_bucket'] = pd.cut(
        predictions_df['actual'],
        bins=[0, 5000, 10000, 20000, float('inf')],
        labels=['$0-5k', '$5-10k', '$10-20k', '$20k+']
    )
    
    for bucket in predictions_df['ltv_bucket'].unique():
        if pd.isna(bucket):
            continue
        bucket_data = predictions_df[predictions_df['ltv_bucket'] == bucket]
        mae = mean_absolute_error(bucket_data['actual'], bucket_data['predicted'])
        mape = (abs(bucket_data['actual'] - bucket_data['predicted']) / bucket_data['actual']).mean() * 100
        
        print(f"   {bucket}: MAE=${mae:,.0f}, MAPE={mape:.1f}%")
    
    return best_model, best_model_name

def generate_predictions_for_new_customers(model, feature_names):
    """
    STEP 5: Generate Predictions for New Customers
    
    Demonstrate how to use the model in production:
    - Create example new customer profiles
    - Generate LTV predictions
    - Make business recommendations
    """
    print("\n" + "="*80)
    print("EXAMPLE: PREDICTING LTV FOR NEW CUSTOMERS")
    print("="*80)
    
    # Create example customers with different profiles
    examples = [
        {
            'profile': 'High-Value Referral',
            'acquisition_channel': 'Referral',
            'initial_plan': 'Enterprise',
            'initial_seats': 30,
            'latent_quality_score': 0.85
        },
        {
            'profile': 'Mid-Market Content Lead',
            'acquisition_channel': 'Content/SEO',
            'initial_plan': 'Pro',
            'initial_seats': 15,
            'latent_quality_score': 0.70
        },
        {
            'profile': 'Small Business Paid Social',
            'acquisition_channel': 'Paid Social',
            'initial_plan': 'Basic',
            'initial_seats': 5,
            'latent_quality_score': 0.50
        }
    ]
    
    print("\n📋 New Customer LTV Predictions:\n")
    
    for example in examples:
        # Create feature vector
        example_df = pd.DataFrame([example])
        
        # One-hot encode to match training features
        example_encoded = pd.get_dummies(
            example_df[['acquisition_channel', 'initial_plan', 'initial_seats', 'latent_quality_score']],
            columns=['acquisition_channel', 'initial_plan']
        )
        
        # Ensure all training features are present
        for col in feature_names:
            if col not in example_encoded.columns:
                example_encoded[col] = 0
        
        # Reorder to match training
        example_encoded = example_encoded[feature_names]
        
        # Predict
        predicted_ltv = model.predict(example_encoded)[0]
        
        # Business recommendations
        if predicted_ltv > 15000:
            recommendation = "🟢 HIGH VALUE - Fast-track, assign dedicated CSM"
            max_cac = predicted_ltv / 3
        elif predicted_ltv > 8000:
            recommendation = "🟡 MEDIUM VALUE - Standard onboarding"
            max_cac = predicted_ltv / 4
        else:
            recommendation = "🔴 LOW VALUE - Self-service only"
            max_cac = predicted_ltv / 5
        
        print(f"   {example['profile']}:")
        print(f"      Channel: {example['acquisition_channel']}, Plan: {example['initial_plan']}, Seats: {example['initial_seats']}")
        print(f"      Quality Score: {example['latent_quality_score']:.2f}")
        print(f"      Predicted LTV: ${predicted_ltv:,.0f}")
        print(f"      Max Allowable CAC: ${max_cac:,.0f}")
        print(f"      Recommendation: {recommendation}")
        print()

def save_model_and_results(model, model_name, feature_names):
    """
    STEP 6: Save Model for Production Use
    
    Save the trained model and metadata for deployment.
    """
    import pickle
    
    output_dir = PROCESSED_DATA_PATH
    
    # Save model
    model_path = os.path.join(output_dir, 'ltv_prediction_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'model_name': model_name,
            'feature_names': feature_names
        }, f)
    
    print("\n" + "="*80)
    print("✅ MODEL SAVED FOR PRODUCTION")
    print("="*80)
    print(f"   Model file: {model_path}")
    print(f"   Model type: {model_name}")
    print(f"   Features: {len(feature_names)}")

def main():
    """
    Main execution flow
    """
    # Step 1: Prepare data
    modeling_data = prepare_training_data()
    
    # Step 2: Engineer features
    X, y, feature_names = engineer_features(modeling_data)
    
    # Step 3: Split data (80% train, 20% test)
    print("\n[Step 3] Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Training set: {len(X_train):,} customers")
    print(f"   ✓ Test set: {len(X_test):,} customers")
    
    # Step 4: Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Evaluate best model
    best_model, best_model_name = evaluate_best_model(
        results, X_test, y_test, feature_names
    )
    
    # Step 6: Example predictions
    generate_predictions_for_new_customers(best_model, feature_names)
    
    # Step 7: Save model
    save_model_and_results(best_model, best_model_name, feature_names)
    
    print("\n" + "="*80)
    print("🎉 PREDICTIVE LTV MODEL COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("   1. Integrate model into signup flow")
    print("   2. Set CAC limits by predicted LTV segment")
    print("   3. Route high-value customers to priority onboarding")
    print("   4. Monitor actual vs predicted LTV over time")
    print("   5. Retrain model quarterly with new data")

if __name__ == '__main__':
    main()