import argparse
import os
import pickle
import re
import statsmodels.api as sm
import pandas as pd
import joblib

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



### ------------------ Prepare Data Functions ------------------ ###


def stepwise_feature_selection(X, y, significance_threshold=0.05):
    """ Performs stepwise regression to select features before applying LDA """
    included = list(X.columns)
    
    while True:
        X_train_sm = sm.add_constant(X[included])  # Add intercept
        model = sm.Logit(y, X_train_sm).fit(disp=False)
        p_values = model.pvalues[1:]  # Exclude intercept
        max_pval = p_values.max()
        
        # Backward Elimination: Remove least significant feature
        if max_pval > significance_threshold:
            worst_feature = p_values.idxmax()
            included.remove(worst_feature)
        else:
            break  # Stop if no more features to remove

    return included  # Return selected features







def prepare_data_logistic(activity_file, customer_file, complaints_file):
    """Prepares data for Logistic Regression (One-Hot Encoding Required)."""
    activity_data = pd.read_csv(activity_file)
    customer_data = pd.read_csv(customer_file)
    complaints = pd.read_excel(complaints_file)

    # Convert dates
    customer_data["birth_date"] = pd.to_datetime(customer_data["birth_date"])
    customer_data["join_date"] = pd.to_datetime(customer_data["join_date"])

    # Feature Engineering
    customer_data["age"] = (pd.to_datetime("today") - customer_data["birth_date"]).dt.days // 365
    customer_data["tenure"] = (pd.to_datetime("today") - customer_data["join_date"]).dt.days // 30  

    # Aggregate activity data
    activity_agg = activity_data.groupby("customer_id")[["data_usage", "phone_usage", "use_app"]].sum().reset_index()
    merged = customer_data.merge(activity_agg, on="customer_id", how="left")

    # Apply One-Hot Encoding (ONLY for Logistic Regression)
    merged = pd.get_dummies(merged, columns=['plan_type'], dtype=int)
    complaints_encoded = pd.get_dummies(complaints, columns=['Category'], dtype=int)

    # Merge complaints
    complaints_encoded = complaints_encoded.rename(columns={'Customer_ID': 'customer_id'})
    merged = pd.merge(merged, complaints_encoded, on='customer_id', how='left')

    merged['Complaint'] = merged['Complaint'].notna().astype(int)
    merged.fillna(0, inplace=True)

    return merged


def prepare_data_lda(activity_file, customer_file, complaints_file):
    """Prepares data for LDA (Label Encoding Required)."""
    activity_data = pd.read_csv(activity_file)
    customer_data = pd.read_csv(customer_file)
    complaints = pd.read_excel(complaints_file)

    # Convert dates
    customer_data["birth_date"] = pd.to_datetime(customer_data["birth_date"])
    customer_data["join_date"] = pd.to_datetime(customer_data["join_date"])

    # Feature Engineering
    customer_data["age"] = (pd.to_datetime("today") - customer_data["birth_date"]).dt.days // 365
    customer_data["tenure"] = (pd.to_datetime("today") - customer_data["join_date"]).dt.days // 30  

    # Aggregate activity data
    activity_agg = activity_data.groupby("customer_id")[["data_usage", "phone_usage", "use_app"]].sum().reset_index()
    merged = customer_data.merge(activity_agg, on="customer_id", how="left")

    # Merge complaints
    complaints = complaints.rename(columns={'Customer_ID': 'customer_id'})
    merged = pd.merge(merged, complaints, on='customer_id', how='left')

    merged['Complaint'] = merged['Complaint'].notna().astype(int)
    merged.fillna(0, inplace=True)

    # Label Encoding (LDA requires numeric categorical features)
    categorical_cols = ['plan_type', 'Category']
    le = LabelEncoder()

    for col in categorical_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)  # 🔹 Convert to string before encoding
            merged[col] = le.fit_transform(merged[col])  # Apply LabelEncoder

    return merged


def prepare_data_ada(activity_file, customer_file, complaints_file):
    """Prepares data for AdaBoost (Label Encoding Required) and returns encoded columns."""
    activity_data = pd.read_csv(activity_file)
    customer_data = pd.read_csv(customer_file)
    complaints = pd.read_excel(complaints_file)

    # Convert dates
    customer_data["birth_date"] = pd.to_datetime(customer_data["birth_date"])
    customer_data["join_date"] = pd.to_datetime(customer_data["join_date"])

    # Feature Engineering
    customer_data["age"] = (pd.to_datetime("today") - customer_data["birth_date"]).dt.days // 365
    customer_data["tenure"] = (pd.to_datetime("today") - customer_data["join_date"]).dt.days // 30  

    # Aggregate activity data
    activity_agg = activity_data.groupby("customer_id")[["data_usage", "phone_usage", "use_app"]].sum().reset_index()
    merged = customer_data.merge(activity_agg, on="customer_id", how="left")

    # Merge complaints
    complaints = complaints.rename(columns={'Customer_ID': 'customer_id'})
    merged = pd.merge(merged, complaints, on='customer_id', how='left')

    merged['Complaint'] = merged['Complaint'].notna().astype(int)
    merged.fillna(0, inplace=True)

    # Label Encoding (Only AdaBoost needs to return encoded_columns)
    categorical_cols = ['plan_type', 'Category']
    encoded_columns = {}
    le = LabelEncoder()

    for col in categorical_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)  # 🔹 Convert to string before encoding
            merged[col] = le.fit_transform(merged[col])  # Apply LabelEncoder
            encoded_columns[col] = le  # Store encoder

    return merged, encoded_columns



def prepare_data_nlp(customer_file,complaints_file):
    complaints = pd.read_excel(complaints_file)
    complaints.rename(columns={'Customer_ID': 'customer_id'}, inplace=True)
    
    customer_data = pd.read_csv(customer_file)
    customer_data = customer_data[['customer_id', 'churn_in_3mos']]
    df = complaints.merge(customer_data, on='customer_id', how='left')
    
    df_churners = df[df['churn_in_3mos'] == 1]
    df_no_churners = df[df['churn_in_3mos'] == 0]
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # List of negation words to keep
    negation_words = {"not", "no", "never", "none", "nothing", "neither", "nor", "can't", "don't", "won't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "doesn't", "didn't", "couldn't", "shouldn't", "wouldn't", "mustn't"}

    # Remove only stopwords that are NOT in the negation list
    filtered_stop_words = stop_words - negation_words
    
    tokenizer = TweetTokenizer()
    
    # Define regex patterns to remove the standard intro and ending
    intro_pattern = r"Subject: Official Complaint\.\s*The undersigned, customer with code \d+, is submitting an official complaint regarding the service\. For the past few weeks, I have encountered the following issue:\s*"
    middle_pattern = r"Despite reaching out to customer support multiple times, the issue remains unresolved\.?\s*"
    ending_pattern = r"I kindly request that you take immediate action to resolve this issue and provide a definitive solution\.\s*I look forward to your prompt response\."
    
    
    # Function to map NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(word):
        """Map NLTK POS tags to WordNet POS tags for better lemmatization."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if not found
    
    # Function to clean complaints
    def preprocess_text(text):
        text = str(text)  # Ensure text format
        text = re.sub(intro_pattern, "", text, flags=re.IGNORECASE)  # Remove intro
        text = re.sub(middle_pattern, "", text, flags=re.IGNORECASE)   # Remove middle pattern
        text = re.sub(ending_pattern, "", text, flags=re.IGNORECASE)  # Remove ending
        text = text.lower()  # Convert to lowercase
        text = text.split(".")[0]
        text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
        words = tokenizer.tokenize(text)  # Tokenize text into words
        words = [word for word in words if word not in filtered_stop_words]  # Remove stopwords
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]  # Correct lemmatization
        return " ".join(words)
    
    
    # Apply preprocessing to complaints
    for i in [df, df_churners]:
        i["Complaint"] = i["Complaint"].astype(str).apply(preprocess_text)
        
    # Function to assign categories
    def assign_category(df, column_name, steps, default_value=None):
        df[column_name] = None
        for step, keywords in steps.items():
            mask = df[column_name].isna() & df["Complaint"].str.lower().apply(
                lambda x: any(all(kw in x for kw in (kw_group if isinstance(kw_group, list) else [kw_group])) for kw_group in keywords)
            )
            df.loc[mask, column_name] = step
    
        # Assign default value to remaining rows if specified
        if default_value is not None:
            df.loc[df[column_name].isna(), column_name] = default_value
    
        return df
    
      # Define keywords

    # Unauthorized charges
    unauthorized_charges = ["authorize", "use", "never sign", "charge", "request"]          # 1.1

    # Promotions not applied
    promotions_not_applied = ["promotion"]                                                  # 1.4

    # Promotions not applied to new joiners that signed up because of promotion
    promotions_new_joiners = [["apply", "sign"], "first", ["discount", "sign"]]             # 1.2

    # Misleading promotional offers
    misleading_promotional_offers = [["mislead", "promotion"]]                              # 1.3

    # Unstable Connection/ internet drops
    unstable_connection = ["internet connection", "internet service", "unstable"]           # 1.7

    # Slow internet speed for plan type
    slow_internet_speed = [["speed", "slow"], "high speed", "premium", "pay", "upgrade"]    # 1.5

    # Phone service unreliable
    phone_service_unreliable = [["phone service", "disconnect"]]                            # 1.6

    # Service unreliable
    service_unreliable = ["service", "connection", "interrupt", "disruption"]               # 1.8

    # Problem resolved?
    not_resolved = []                                                                       # 2.2
    resolved_but_slow = ["restore", "long", ["slow", "resolve"]]                            # 2.1

    # Short-term or not defined
    short_term_or_not_defined = []                                                          # 3.2

    # Long-term
    long_term = ["hour", "day", "long", "several hour", "several day", "several week"]      # 3.1
    
    
    

    
    
    
    # Define step orders
    problem_steps = {
        "Unauthorized charges": unauthorized_charges,  # 1.1
        "Promotions not applied to new joiners": promotions_new_joiners,  # 1.2
        "Misleading promotional offers": misleading_promotional_offers,  # 1.3
        "Promotions not applied": promotions_not_applied,  # 1.4
        "Slow internet speed for plan type": slow_internet_speed,  # 1.5
        "Phone service unreliable": phone_service_unreliable,  # 1.6
        "Unstable Connection/ internet drops": unstable_connection,  # 1.7
        "Service unreliable": service_unreliable,  # 1.8
    }

    resolved_steps = {
        "Resolved but slow": resolved_but_slow,  # 2.1
    }

    duration_steps = {
        "Long-term": long_term,  # 3.1
    }
        
      # Apply categorization
    df_churners = assign_category(df_churners, "Problem", problem_steps)
    df_churners = assign_category(df_churners, "Problem Resolved", resolved_steps, default_value="Not resolved")
    df_churners = assign_category(df_churners, "Duration", duration_steps, default_value="Short-term")
        
    return df_churners



def train_model_nlp(activity_file, customer_file, complaints_file, output_model):
    
  
    
    activity_data = pd.read_csv(activity_file)
    
    complaints = pd.read_excel(complaints_file)
    
    customer_data = pd.read_csv(customer_file)
    
    # Convert 'month' to datetime for proper analysis
    activity_data['month'] = pd.to_datetime(activity_data['month'], format="%d/%m/%Y")
    
    # Convert dates to datetime format
    customer_data["birth_date"] = pd.to_datetime(customer_data["birth_date"])
    customer_data["join_date"] = pd.to_datetime(customer_data["join_date"])
    
    # Create additional features
    customer_data["age"] = (pd.to_datetime("today") - customer_data["birth_date"]).dt.days // 365
    customer_data["tenure"] = (pd.to_datetime("today") - customer_data["join_date"]).dt.days // 30  # Months
    
    
    # Aggregate activity_data by customer_id
    activity_agg = activity_data.groupby("customer_id")[["data_usage", "phone_usage", "use_app"]].sum().reset_index()
    
    
    # Merge with customer_data
    agg_merged_data = customer_data.merge(activity_agg, on="customer_id", how="left")
    
    # Load the categorized complaints data
    complaints_df = prepare_data_nlp(customer_file,complaints_file)
    
    # Merge with agg_merged_data on 'customer_id', keeping only matching rows
    merged_all_complaints_df = agg_merged_data.merge(complaints_df, on="customer_id", how="inner")
    
    # Drop the specified columns from the merged dataframe
    columns_to_drop = ["Duration", "Problem Resolved", "Category", "Complaint", "birth_date", "join_date", "churn_in_3mos_y"]
    merged_all_complaints_df = merged_all_complaints_df.drop(columns=columns_to_drop, errors='ignore')
    merged_all_complaints_df.rename(columns={'churn_in_3mos_x': 'churn_in_3mos'}, inplace=True)
    
    
    # Make a copy of the dataset to avoid modifying the original
    df = merged_all_complaints_df.copy()

    # Drop customer_id as it is not useful for classification
    df.drop(columns=['customer_id'], inplace=True, errors='ignore')
    
        # Make a copy of the dataset to avoid modifying the original
    df = merged_all_complaints_df.copy()

    # Drop customer_id as it is not useful for classification
    df.drop(columns=['customer_id'], inplace=True, errors='ignore')
    
    
    
    # Convert categorical features using Label Encoding
    label_encoders = {}
    categorical_cols = ['plan_type']  # Add more if necessary

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoder for later decoding

    # Encode 'Problem' column (target variable)
    le_problem = LabelEncoder()
    df['Problem'] = le_problem.fit_transform(df['Problem'])
    
    # Split dataset into features (X) and target variable (y)
    X = df.drop(columns=['Problem'])
    y = df['Problem']
    


    # Split into training and test set (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    
    y_pred = clf.predict(X_test)
    
    
    output_dir = os.path.dirname(output_model)  # Extract directory from the output path
    os.makedirs(output_dir, exist_ok=True)
    
    # ✅ Save the trained model to the specified path
    with open(output_model, "wb") as model_file:
        pickle.dump(clf, model_file)

    print(f"✅ Model saved to {output_model}")
    
    
  
    
    
    

### ------------------ Model Training Functions ------------------ ###

def train_model_logistic(activity_file, customer_file, complaints_file, output_model, output_scaler, output_features):
    """Trains Logistic Regression Model."""
    data = prepare_data_logistic(activity_file, customer_file, complaints_file)

    X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date', 'Complaint'])
    y = data['churn_in_3mos']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    selected_features = stepwise_feature_selection(X_train, y_train)
    
    

    log_reg = LogisticRegression()
    log_reg.fit(X_train[selected_features], y_train)




    y_pred = log_reg.predict(X_test[selected_features])
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    joblib.dump(log_reg, output_model)
    joblib.dump(scaler, output_scaler)
    joblib.dump(selected_features, output_features)


def train_model_lda(activity_file, customer_file, complaints_file, output_model, output_scaler, output_features):
    """Trains LDA Model."""
    data = prepare_data_lda(activity_file, customer_file, complaints_file)

    X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date'])
    y = data['churn_in_3mos']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    selected_features = stepwise_feature_selection(X_train, y_train)

    lda = LDA()
    lda.fit(X_train[selected_features], y_train)

    y_pred = lda.predict(X_test[selected_features])
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    joblib.dump(lda, output_model)
    
    joblib.dump(scaler, output_scaler)
    
    joblib.dump(selected_features, output_features)


def train_model_ada(activity_file, customer_file, complaints_file, output_model, output_features):
    """Trains AdaBoost (Saves Encoders)."""
    data, encoded_columns = prepare_data_ada(activity_file, customer_file, complaints_file)
    
    X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date'])
    y = data['churn_in_3mos']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    selected_features = stepwise_feature_selection(X_train, y_train)

    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
    ada.fit(X_train[selected_features], y_train)
    
    y_pred = ada.predict(X_test[selected_features])
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    joblib.dump(ada, output_model)
    joblib.dump(selected_features, output_features)

### ------------------ Fixed Main Script ------------------ ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a customer churn prediction model.")

    parser.add_argument("--activity", required=True, help="Path to activity data CSV")
    parser.add_argument("--customer", required=True, help="Path to customer data CSV")
    parser.add_argument("--complaints", required=True, help="Path to complaints Excel file")
    parser.add_argument("--model_type", required=True, choices=["logistic", "lda", "adaboost", "nlp"], help="Type of model to train")
    parser.add_argument("--output_model", default="model.pkl", help="Path to save trained model")
    parser.add_argument("--output_scaler", default="scaler.pkl", help="Path to save scaler (only for logistic regression and LDA)")
    parser.add_argument("--output_features", default="feature_names.pkl", help="Path to save selected features (only for AdaBoost)")

    args = parser.parse_args()

    if args.model_type == "logistic":
        train_model_logistic(args.activity, args.customer, args.complaints, args.output_model, args.output_scaler, args.output_features)
    elif args.model_type == "lda":
        train_model_lda(args.activity, args.customer, args.complaints, args.output_model, args.output_scaler, args.output_features)
    elif args.model_type == "adaboost":
        train_model_ada(args.activity, args.customer, args.complaints, args.output_model, args.output_features)
    elif args.model_type=="nlp":
        train_model_nlp(args.activity, args.customer, args.complaints, args.output_model)

    print(f"Training complete! Model saved to {args.output_model}.")
