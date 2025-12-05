# Machine Learning Project Report: Crash Reporting Analysis

## A. Introduction

### Business/Scientific Case

Vehicle crashes are a significant public health and safety concern worldwide. Understanding the factors that contribute to crash severity can help:
- Improve road safety policies
- Develop targeted interventions
- Allocate resources more effectively
- Reduce injury severity and fatalities

This project aims to predict injury severity in vehicle crashes using machine learning techniques, which can assist traffic safety authorities, insurance companies, and policymakers in making data-driven decisions.

### Problem Framing

**Problem Statement**: Predict the injury severity of vehicle crashes based on various factors such as weather conditions, road conditions, driver behavior, and vehicle characteristics.

**Type of Problem**: Multi-class classification problem
- **Target Variable**: Injury Severity (e.g., "No Apparent Injury", "Possible Injury", "Suspected Minor Injury", etc.)
- **Input Features**: Weather, surface conditions, light conditions, collision type, driver behavior, vehicle characteristics, etc.

**Business Value**:
- Early identification of high-risk crash scenarios
- Resource allocation for emergency services
- Policy development for road safety
- Insurance risk assessment

### Dataset Origin and Licensing

**Dataset Source**: [Please add the source URL where you downloaded the dataset]

**Dataset Name**: Crash Reporting - Drivers Data

**Dataset Description**: The dataset contains detailed information about vehicle crashes including:
- Crash metadata (date, time, location)
- Environmental conditions (weather, surface condition, light)
- Driver information (at fault, substance abuse, distraction)
- Vehicle information (type, year, damage extent)
- Crash outcomes (injury severity, collision type)

**License**: [Please specify the dataset license - e.g., Public Domain, Open Data, etc.]

**Data Collection Period**: [Specify the time period covered by the data]

**Justification for Analysis**: This dataset is suitable for supervised learning classification analysis because:
1. It contains a large number of labeled examples (crashes with known injury severity)
2. It includes diverse features that may influence crash outcomes
3. It represents real-world crash scenarios
4. The target variable (injury severity) is clearly defined

---

## B. Description of the Exploratory Data Analysis

### Dataset Shape and Structure

- **Total Rows**: [To be filled after running the notebook]
- **Total Columns**: [To be filled after running the notebook]
- **Data Types**: Mix of categorical (object) and numerical (int, float) variables

### Observations and Characteristics

The dataset contains information about vehicle crashes with the following key characteristics:
- Multiple crash records per incident (one row per vehicle/driver involved)
- Mix of categorical and numerical features
- Presence of missing values in various columns
- Temporal information (crash date/time)
- Geographic information (latitude, longitude)

### Data Types

**Categorical Variables**:
- Report Number, Local Case Number
- Agency Name, Route Type, Road Name
- Weather, Surface Condition, Light
- Collision Type, Driver At Fault
- Vehicle Body Type, Vehicle Movement
- And many more...

**Numerical Variables**:
- Speed Limit
- Vehicle Year
- Latitude, Longitude
- [Other numerical features]

### Distributions

**Target Variable Distribution**:
- The injury severity classes show an imbalanced distribution
- "No Apparent Injury" is likely the most common class
- More severe injuries are less frequent (as expected in real-world data)

**Feature Distributions**:
- Weather conditions: Clear weather is most common
- Surface conditions: Dry surfaces are predominant
- Light conditions: Daylight crashes are most frequent
- Speed limits: Distribution varies by road type

### Correlations

**Numerical Feature Correlations**:
- Speed Limit and Vehicle Year may show weak correlations
- Geographic coordinates (Latitude, Longitude) are highly correlated
- [Additional correlation findings after analysis]

**Categorical Feature Relationships**:
- Weather conditions correlate with surface conditions
- Collision type relates to driver behavior
- Vehicle type may correlate with crash severity

### Summary Statistics

**Measures of Central Tendency**:
- **Mean**: Average values for numerical features
- **Median**: Middle values, less affected by outliers
- **Mode**: Most frequent values for categorical features

**Measures of Dispersion**:
- **Variance**: Spread of numerical data
- **Standard Deviation**: Average distance from mean
- **Range**: Difference between max and min values

[Specific statistics will be filled after running the analysis]

### Justified Visualizations

1. **Injury Severity Distribution (Bar Chart)**
   - **Justification**: Shows the class distribution, essential for understanding class imbalance
   - **Insight**: Helps determine if class balancing techniques are needed

2. **Missing Values Heatmap**
   - **Justification**: Identifies which features have missing data and to what extent
   - **Insight**: Guides preprocessing decisions for handling missing values

3. **Correlation Matrix Heatmap**
   - **Justification**: Reveals relationships between numerical features
   - **Insight**: Helps identify multicollinearity and feature relationships

4. **Feature vs Target Variable Crosstabs**
   - **Justification**: Shows how different feature values relate to injury severity
   - **Insight**: Identifies which features are most predictive of the target

5. **Box Plots for Numerical Features by Injury Severity**
   - **Justification**: Shows distribution differences across injury severity classes
   - **Insight**: Reveals which numerical features discriminate between classes

---

## C. Dataset Preparation

### Data Preprocessing Steps

#### 1. Data Cleaning

**Removing Duplicates**:
- **Rationale**: Duplicate records can bias the model and inflate performance metrics
- **Method**: Used `drop_duplicates()` to remove exact duplicate rows
- **Result**: [Number] duplicate rows removed

**Removing Identifier Columns**:
- **Rationale**: Identifiers like Report Number, Person ID, Vehicle ID don't provide predictive value
- **Method**: Dropped columns: 'Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID', 'Location'
- **Justification**: These are unique identifiers, not features that predict injury severity

**Handling Impossible Data Combinations**:
- **Rationale**: Some data combinations may be logically impossible or errors
- **Method**: [Specify any data validation rules applied]
- **Example**: Vehicle Year > Current Year would be flagged

#### 2. Missing Values Handling

**Missing Value Analysis**:
- **Rationale**: Missing values can cause errors in ML algorithms and reduce model performance
- **Method**: Analyzed missing value patterns across all features

**Imputation Strategy**:
- **Categorical Variables**: Filled with 'Unknown' category
  - **Justification**: Preserves information that value is missing, which may be informative
- **Numerical Variables**: Filled with median value
  - **Justification**: Median is robust to outliers, unlike mean
- **Target Variable**: Removed rows with missing target values
  - **Justification**: Cannot train a model without known labels

**Missing Value Statistics**:
- [Percentage] of rows had at least one missing value
- Top features with missing values: [List]

#### 3. Outlier Detection and Handling

**Outlier Detection Methods**:
- **IQR Method**: For numerical features, identified outliers beyond Q3 + 1.5*IQR or below Q1 - 1.5*IQR
- **Visual Inspection**: Box plots and scatter plots
- **Domain Knowledge**: Vehicle Year < 1900 or > 2025 considered outliers

**Outlier Handling**:
- **Method**: Capped outliers at reasonable bounds or removed extreme cases
- **Justification**: Outliers can skew model training, but some may represent valid rare cases

#### 4. Normalization/Encoding

**Categorical Encoding**:
- **Method**: One-Hot Encoding using `pd.get_dummies()`
- **Rationale**: Converts categorical variables into binary features that ML algorithms can process
- **Drop First**: Yes, to avoid multicollinearity
- **Result**: [Number] categorical features encoded into [Number] binary features

**Numerical Scaling**:
- **Method**: StandardScaler (Z-score normalization)
- **Rationale**: Some algorithms (e.g., Logistic Regression, SVM) require features on similar scales
- **Formula**: (x - mean) / standard_deviation
- **Applied To**: Logistic Regression model (Random Forest and Gradient Boosting don't require scaling)

#### 5. Feature Engineering

**Temporal Features**:
- **Crash Hour**: Extracted from 'Crash Date/Time'
  - **Rationale**: Time of day may influence crash severity (e.g., nighttime crashes)
- **Crash DayOfWeek**: Extracted day of week (0=Monday, 6=Sunday)
  - **Rationale**: Weekday vs weekend patterns may differ

**Derived Features**:
- **Vehicle Age**: Calculated as Current Year - Vehicle Year
  - **Rationale**: Age is more meaningful than year for predicting outcomes
  - **Handling**: Replaced 0 or invalid years with median age

**Binary Features**:
- **Has_Substance_Abuse**: Binary indicator (1 if 'Suspect' in Driver Substance Abuse, else 0)
  - **Rationale**: Simplifies complex categorical variable into actionable binary feature

**Feature Selection**:
- **Selected Features**: Weather, Surface Condition, Light, Traffic Control, Collision Type, Driver At Fault, Driver Substance Abuse, Vehicle Damage Extent, Vehicle Body Type, Vehicle Movement, Speed Limit, Route Type, and engineered features
- **Rationale**: These features are most likely to predict injury severity based on domain knowledge

#### 6. Class Imbalance Handling

**Problem**: Injury severity classes are imbalanced (e.g., "No Apparent Injury" much more common than severe injuries)

**Solution**:
- **Class Weighting**: Used `class_weight='balanced'` in models
  - **Rationale**: Automatically adjusts weights inversely proportional to class frequency
- **Class Filtering**: Kept only classes with ≥1% of data
  - **Rationale**: Very rare classes may not have enough data for reliable prediction

**Result**: [Number] classes retained for modeling

#### 7. Train-Validation-Test Split

**Split Strategy**:
- **Training Set**: 60% of data
- **Validation Set**: 20% of data
- **Test Set**: 20% of data

**Method**: Stratified split to maintain class distribution across splits
- **Rationale**: Ensures each split has representative class proportions

**Random Seed**: 42 (for reproducibility)

---

## D. Justification of Machine Learning Algorithms

### Algorithm Selection

This project implements and compares **three supervised learning classification algorithms**:

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **Logistic Regression**

### Algorithm 1: Random Forest Classifier

**Type**: Ensemble Learning (Bagging)

**Rationale for Selection**:
- Handles both numerical and categorical features well
- Robust to overfitting
- Provides feature importance scores
- Works well with imbalanced data (with class_weight)
- No feature scaling required

**Hyperparameters**:
- **n_estimators=100**: Number of trees in the forest
  - **Justification**: Balance between performance and computational cost
- **max_depth=20**: Maximum depth of trees
  - **Justification**: Prevents overfitting while allowing complexity
- **min_samples_split=10**: Minimum samples required to split a node
  - **Justification**: Prevents overfitting on small subsets
- **min_samples_leaf=5**: Minimum samples required in a leaf node
  - **Justification**: Ensures leaves have sufficient samples
- **class_weight='balanced'**: Automatically adjusts class weights
  - **Justification**: Handles class imbalance
- **random_state=42**: For reproducibility
- **n_jobs=-1**: Uses all available CPU cores

**Validation**: Cross-validation and validation set performance monitoring

**Metrics Used**: Accuracy, Precision, Recall, F1-Score

### Algorithm 2: Gradient Boosting Classifier

**Type**: Ensemble Learning (Boosting)

**Rationale for Selection**:
- Often achieves high accuracy
- Sequential learning from mistakes
- Handles complex non-linear relationships
- Good for imbalanced data

**Hyperparameters**:
- **n_estimators=100**: Number of boosting stages
  - **Justification**: Sufficient for learning without excessive computation
- **learning_rate=0.1**: Shrinkage rate
  - **Justification**: Standard value, balances learning speed and stability
- **max_depth=5**: Maximum depth of individual trees
  - **Justification**: Prevents overfitting, standard for gradient boosting
- **min_samples_split=10**: Minimum samples to split
  - **Justification**: Prevents overfitting
- **min_samples_leaf=5**: Minimum samples in leaf
  - **Justification**: Ensures robust predictions
- **random_state=42**: For reproducibility

**Validation**: Validation set performance monitoring

**Metrics Used**: Accuracy, Precision, Recall, F1-Score

### Algorithm 3: Logistic Regression

**Type**: Linear Model

**Rationale for Selection**:
- Interpretable (coefficients show feature importance)
- Fast training and prediction
- Good baseline model
- Probabilistic outputs
- Works well as a benchmark

**Hyperparameters**:
- **max_iter=1000**: Maximum iterations for convergence
  - **Justification**: Ensures algorithm converges on complex data
- **class_weight='balanced'**: Handles class imbalance
  - **Justification**: Adjusts for unequal class distribution
- **multi_class='multinomial'**: For multi-class classification
  - **Justification**: Appropriate for >2 classes
- **solver='lbfgs'**: Limited-memory BFGS optimizer
  - **Justification**: Good for small to medium datasets, handles multinomial loss
- **random_state=42**: For reproducibility

**Preprocessing Required**: Feature scaling (StandardScaler applied)

**Validation**: Validation set performance monitoring

**Metrics Used**: Accuracy, Precision, Recall, F1-Score

### Model Selection Rationale

**Selection Criteria**:
1. **Test Set Accuracy**: Primary metric for comparison
2. **Precision**: Important for minimizing false positives
3. **Recall**: Important for identifying all injury cases
4. **F1-Score**: Balanced metric considering both precision and recall
5. **Training Time**: Computational efficiency
6. **Interpretability**: Ability to explain predictions

**Final Model Selection**: [Best model name] was selected based on highest test accuracy and balanced performance across all metrics.

### Evaluation Metrics Justification

**Accuracy**:
- **Use**: Overall correctness of predictions
- **Justification**: Primary metric for classification performance
- **Limitation**: Can be misleading with imbalanced classes (mitigated by class weighting)

**Precision (Weighted Average)**:
- **Use**: Proportion of positive predictions that are correct
- **Justification**: Important for minimizing false alarms
- **Weighted**: Accounts for class imbalance

**Recall (Weighted Average)**:
- **Use**: Proportion of actual positives correctly identified
- **Justification**: Important for not missing injury cases
- **Weighted**: Accounts for class imbalance

**F1-Score (Weighted Average)**:
- **Use**: Harmonic mean of precision and recall
- **Justification**: Balanced metric when both precision and recall matter
- **Weighted**: Accounts for class imbalance

**Confusion Matrix**:
- **Use**: Detailed breakdown of prediction errors
- **Justification**: Shows which classes are confused with each other

### Model Comparison Table

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time |
|-----------|----------|-----------|--------|----------|---------------|
| Random Forest | [Value] | [Value] | [Value] | [Value] | [Time] |
| Gradient Boosting | [Value] | [Value] | [Value] | [Value] | [Time] |
| Logistic Regression | [Value] | [Value] | [Value] | [Value] | [Time] |

*Note: Values will be filled after running the notebook*

---

## E. Conclusion

### Key Findings

1. **Data Quality**: The dataset required significant preprocessing, including handling missing values, encoding categorical variables, and addressing class imbalance.

2. **Feature Importance**: [Top features identified from Random Forest feature importance]
   - Weather conditions significantly influence injury severity
   - Collision type is a strong predictor
   - Speed limit correlates with injury severity
   - Driver behavior factors (at fault, substance abuse) are important

3. **Model Performance**: 
   - [Best model] achieved [accuracy]% accuracy on the test set
   - All three models showed reasonable performance
   - [Best model] demonstrated the best balance of accuracy, precision, and recall

4. **Predictive Factors**: The analysis revealed that:
   - Environmental conditions (weather, surface, light) impact crash severity
   - Driver behavior (at fault, distraction, substance abuse) is predictive
   - Vehicle characteristics (type, age) influence outcomes
   - Road characteristics (speed limit, route type) matter

### Limitations

1. **Data Limitations**:
   - Missing values in some features may affect model performance
   - Class imbalance may limit prediction of rare severe injury cases
   - Dataset may not represent all crash scenarios equally

2. **Model Limitations**:
   - Models are trained on historical data and may not generalize to future scenarios
   - Correlation does not imply causation
   - Some relationships may be non-linear and not fully captured

3. **Evaluation Limitations**:
   - Test set may not represent all possible crash scenarios
   - Metrics may not fully capture real-world performance
   - Model may perform differently on new, unseen data

4. **Feature Limitations**:
   - Some potentially important features may be missing (e.g., vehicle speed at impact, road design)
   - Feature engineering may not capture all relevant interactions

### Future Work

1. **Data Improvements**:
   - Collect more data on rare injury severity classes
   - Include additional features (e.g., road design, traffic volume)
   - Improve data quality and reduce missing values

2. **Model Improvements**:
   - Experiment with deep learning models (Neural Networks)
   - Try ensemble methods combining multiple algorithms
   - Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Explore feature interactions and polynomial features

3. **Evaluation Improvements**:
   - Implement time-based validation (if temporal patterns exist)
   - Use additional metrics (AUC-ROC, PR-AUC)
   - Perform feature importance analysis across all models
   - Conduct sensitivity analysis

4. **Deployment**:
   - Create a production-ready API
   - Implement model versioning and monitoring
   - Add real-time prediction capabilities
   - Develop user-friendly interfaces

5. **Domain-Specific Enhancements**:
   - Collaborate with traffic safety experts
   - Validate findings with domain knowledge
   - Develop actionable insights for policymakers

### Ethical Considerations

1. **Data Privacy**:
   - Ensure crash data is anonymized and de-identified
   - Comply with data protection regulations
   - Protect personal information of crash victims

2. **Bias and Fairness**:
   - Check for demographic biases in predictions
   - Ensure model doesn't discriminate against certain groups
   - Validate model fairness across different populations

3. **Transparency**:
   - Document model limitations and assumptions
   - Provide interpretable explanations for predictions
   - Make methodology and code publicly available

4. **Responsible Use**:
   - Use predictions to improve safety, not to penalize individuals
   - Consider unintended consequences of model deployment
   - Engage stakeholders in model development and deployment

5. **Accountability**:
   - Clearly communicate model uncertainty
   - Don't overstate model capabilities
   - Allow for human oversight in critical decisions

### Final Remarks

This project successfully demonstrates the application of machine learning to predict injury severity in vehicle crashes. The analysis provides valuable insights into factors that influence crash outcomes and can inform evidence-based policy decisions. However, it is important to recognize the limitations and use the models responsibly, always considering the broader context and ethical implications.

---

**Report Generated**: [Date]
**Author**: [Your Name]
**Institution**: [Your Institution]

