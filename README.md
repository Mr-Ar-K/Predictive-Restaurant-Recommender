# Predictive Restaurant Recommender

## Overview

A machine learning model that predicts the probability of customers ordering from specific restaurants based on customer profiles, order history, and restaurant information. The model uses LightGBM with advanced feature engineering to deliver accurate restaurant recommendations.

## Problem Statement

Build a recommendation engine to predict what restaurants customers are most likely to order from, given:
- Customer location and demographics
- Restaurant information and characteristics  
- Customer order history and preferences

## Data Structure

```
├── Train/
│   ├── orders.csv              # Historical order transactions
│   ├── train_customers.csv     # Customer profiles and demographics
│   ├── train_locations.csv     # Customer delivery locations
│   └── vendors.csv             # Restaurant/vendor information
├── Test/
│   ├── test_customers.csv      # Test customer profiles
│   ├── test_locations.csv      # Test customer locations
│   └── submission.csv          # Model predictions (generated)
└── model.ipynb                 # Complete ML pipeline
```

## Data Description

### Train Customers
Customer demographic and account information:
- `customer_id`: Unique identifier
- `gender`: Customer gender
- `dob`: Birth year
- `status`, `verified`: Account status
- `language`: Preferred language
- `created_at`, `updated_at`: Account timestamps

### Train Locations  
Customer delivery locations with masked coordinates:
- `customer_id`: Links to customer data
- `location_number`: Location identifier (1, 2, etc.)
- `location_type`: Home, Work, Other, or NA
- `latitude`, `longitude`: Masked coordinates (relative positioning preserved)

### Train Orders
Complete order transaction history:
- `order_id`: Internal order identifier
- `customer_id`: Customer making the order
- `vendor_id`: Restaurant/vendor identifier
- `item_count`: Number of items ordered
- `grand_total`: Total order cost
- **Payment**: `payment_mode`, `Promo_code`, discounts
- **Ratings**: `is_favorite`, `is_rated`, `vendor_rating`, `driver_rating`
- **Logistics**: `deliverydistance`, `preparationtime`, `delivery_time`, timestamps
- `CID X LOC_NUM X VENDOR`: Submission format identifier

### Vendors
Restaurant/vendor information:
- `id`: Unique vendor identifier
- `latitude`, `longitude`: Masked coordinates (same reference frame as customers)
- `vendor_tag_name`: Descriptive tags (cuisine type, features)
- `vendor_category_en`: Restaurant category
- `delivery_charge`, `serving_distance`: Service parameters
- `is_open`, `status`: Operational status

## Solution Pipeline

### 1. Data Processing
- Load and merge all data sources
- Handle missing values and data type conversions
- Create balanced positive/negative training samples

### 2. Feature Engineering
- **Customer Features**: Order frequency, spending patterns, vendor diversity
- **Vendor Features**: Popularity metrics, average ratings, order volumes
- **Interaction Features**: Customer-vendor history, favorites, ratings
- **Location Features**: Distance calculations, location preferences
- **Temporal Features**: Order timing patterns, customer lifetime

### 3. Model Training
- **Algorithm**: LightGBM Gradient Boosting
- **Validation**: 5-fold stratified cross-validation
- **Optimization**: Hyperparameter tuning with Optuna
- **Ensemble**: Multiple models for robust predictions

### 4. Prediction & Output
- Generate probabilities for all customer-location-vendor combinations
- Output format: `CID X LOC_NUM X VENDOR, target`
- Single submission file: `Test/submission.csv`

## Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm optuna
```

### Running the Model
1. Ensure data files are in correct `Train/` and `Test/` directories
2. Open `model.ipynb` in Jupyter or VS Code
3. Run all cells sequentially
4. Final predictions saved to `Test/submission.csv`

### Expected Output
- **Format**: CSV with columns `CID X LOC_NUM X VENDOR`, `target`
- **Size**: ~700K+ predictions covering all test combinations
- **Values**: Probabilities between 0 and 1

## Model Performance

- **Algorithm**: LightGBM with ensemble averaging
- **Features**: 30+ engineered features
- **Validation**: Cross-validated AUC score
- **Optimization**: Automated hyperparameter tuning

## Key Features

- ✅ **Minimal Codebase**: Clean, production-ready notebook
- ✅ **Advanced Features**: Customer behavior, vendor popularity, interaction history
- ✅ **Robust Training**: Cross-validation, hyperparameter optimization, ensemble methods
- ✅ **Scalable Pipeline**: Efficient data processing and prediction generation
- ✅ **Single Output**: Only generates required submission file

## Repository Structure

```
Predictive-Restaurant-Recommender/
├── README.md           # This file
├── LICENSE            # Project license
├── model.ipynb        # Complete ML pipeline
├── Train/             # Training data directory
└── Test/              # Test data and output directory
```
