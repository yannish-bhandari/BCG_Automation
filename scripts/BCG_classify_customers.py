import pandas as pd
import joblib
import argparse
import os
import pickle
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


### ------------------ Prepare Data Functions ------------------ ###
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


### ------------------ Prediction Function ------------------ ###
def predict(activity_file, customer_file, complaints_file, model_type, model_file, scaler_file, features_file):
    """Runs predictions using the selected model type."""

    # Load model
    loaded_object = joblib.load(model_file)

    # If model was saved as a dictionary (AdaBoost), extract encoders and model
    if isinstance(loaded_object, dict) and "adaboost_model" in loaded_object:
        encoded_columns = loaded_object["label_encoders"]
        model = loaded_object["adaboost_model"]
    else:
        encoded_columns = None
        model = loaded_object

    # Select the correct `prepare_data_*` function
    if model_type == "logistic":
        data = prepare_data_logistic(activity_file, customer_file, complaints_file)
        scaler = joblib.load(scaler_file)  # Load scaler
    elif model_type == "lda":
        data = prepare_data_lda(activity_file, customer_file, complaints_file)
        scaler = joblib.load(scaler_file)
    elif model_type == "adaboost":
        data, encoded_columns = prepare_data_ada(activity_file, customer_file, complaints_file)  # ✅ Extract only the DataFrame
        scaler = None  # AdaBoost does not require a scaler
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    feature_names = joblib.load(features_file)

    # **Remove columns based on the training process**
    if model_type == "logistic":
        X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date', 'Complaint'], errors='ignore')
    elif model_type == "lda":
        X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date'], errors='ignore')
    elif model_type == "adaboost":
        X = data.drop(columns=['churn_in_3mos', 'birth_date', 'join_date'], errors='ignore')
    else:
        raise ValueError("Invalid model type")

    # ✅ Step 1: Apply scaling first (on all features present in the dataset)
    if scaler is not None:
        X_scaled = scaler.transform(X)  # This returns a NumPy array

        # ✅ Step 2: Convert back to DataFrame with the original feature names
        X = pd.DataFrame(X_scaled, columns=X.columns)

    # ✅ Step 3: Ensure only the trained features are selected AFTER SCALING
    X = X.loc[:, X.columns.intersection(feature_names)]

    predictions = model.predict(X)

    data["churn_in_3mos"] = predictions  # ✅ Store churn predictions

    return data[["customer_id", "churn_in_3mos"]]


# ##-------------------NLP------------------##
def prepare_data_nlp(customer_file, complaints_file):
    complaints = pd.read_excel(complaints_file)
    complaints.rename(columns={'Customer_ID': 'customer_id'}, inplace=True)

    customer_data = pd.read_csv(customer_file)
    customer_data = customer_data[['customer_id', 'churn_in_3mos']]
    df = complaints.merge(customer_data, on='customer_id', how='left').copy()

    df_churners = df[df['churn_in_3mos'] == 1].copy()
    df_no_churners = df[df['churn_in_3mos'] == 0].copy()
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


def final_pred(activity_file, customer_file, complaints_file, model_type, model_file, nlp_file, scaler_file, features_file, output_file):

    activity_data = pd.read_csv(activity_file)

    complaints = pd.read_excel(complaints_file)

    customer_data = pd.read_csv(customer_file)

    complaints_df = prepare_data_nlp(customer_file, complaints_file)

    predictions = predict(activity_file, customer_file, complaints_file, model_type, model_file, scaler_file, features_file)
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
    columns_to_drop = [ "birth_date", "join_date", "churn_in_3mos"]

    final_df = agg_merged_data.drop(columns=[col for col in columns_to_drop if col in agg_merged_data.columns])

    final_df = final_df.merge(predictions, on="customer_id", how="left")

    label_encoders = {}
    categorical_cols = ['plan_type']  # Add more if necessary

    for col in categorical_cols:
        le = LabelEncoder()
        final_df[col] = le.fit_transform(final_df[col])
        label_encoders[col] = le  # Store encoder for later decoding

    # ✅ Step 2: Drop `customer_id` only if it exists
    if "customer_id" in final_df.columns:
        final_df = final_df.drop(columns=["customer_id"])

    loaded_object = joblib.load(nlp_file)

    model = loaded_object

    # ✅ Step 4: Extract feature names from the trained model
    if hasattr(model, "feature_names_in_"):
        trained_features = model.feature_names_in_.tolist()  # Extract feature names
    else:
        trained_features = list(final_df.columns)  # Use available columns as fallback

    # ✅ Step 5: Ensure `final_df` matches the trained model's feature set (order & missing features)
    final_df = final_df.reindex(columns=trained_features, fill_value=0)

    # ✅ Step 6: Predict using the correctly formatted `final_df`

    # Merge with agg_merged_data on 'customer_id', keeping only matching rows
    merged_all_complaints_df = agg_merged_data.merge(complaints_df, on="customer_id", how="inner")

    columns_to_drop = ["Duration", "Problem Resolved", "Category", "Complaint"]

    merged_all_complaints_df = merged_all_complaints_df.drop(columns=columns_to_drop, errors='ignore')

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

    # Drop birth_date and join_date from agg_merged_data
    agg_merged_data = agg_merged_data.drop(columns=['birth_date', 'join_date'], errors='ignore')

    # Merge with merged_all_complaints_df to add the "Problem" column
    merged_full_df = agg_merged_data.merge(merged_all_complaints_df[['customer_id', 'Problem']], on="customer_id", how="left")

    # Filter out customers who have churned but did not complain (NaN in 'Problem' column)
    filtered_churned_df = merged_full_df[(merged_full_df['churn_in_3mos'] == 1) & (merged_full_df['Problem'].isna())]

    # Make a copy of the filtered dataframe to avoid modifying the original
    df_to_classify = filtered_churned_df.copy()

    # Drop customer_id as it is not useful for classification
    df_to_classify.drop(columns=['customer_id'], inplace=True, errors='ignore')

    # Convert categorical features using Label Encoding (use same encoder as training)
    for col in categorical_cols:
        if col in df_to_classify.columns:
            df_to_classify[col] = label_encoders[col].transform(df_to_classify[col])  # Use existing encoder

    # Drop 'Problem' column before classification
    X_new = df_to_classify.drop(columns=['Problem'], errors='ignore')

    predicted_labels = model.predict(X_new)

    #   Convert numerical predictions back to text labels
    df_to_classify['Problem'] = le_problem.inverse_transform(predicted_labels)

    # Reattach customer_id for reference
    df_to_classify['customer_id'] = filtered_churned_df['customer_id'].values

    df_churners = complaints_df

    # Append df_to_classify to df_churners without losing any rows
    merged_final_df = pd.concat([df_churners, df_to_classify], ignore_index=True)

    # Select required columns, ensuring customer_id is first
    selected_columns = ["customer_id", "Complaint", "churn_in_3mos", "Problem", "Problem Resolved", "Duration"]
    merged_final_df = merged_final_df[selected_columns]

    merged_final_df = merged_final_df.rename(columns={"churn_in_3mos": "Churn_Prediction"})

    cols = list(merged_final_df.columns)  # Get the current column order
    cols.remove("Churn_Prediction")  # Remove the target column from the list
    cols.insert(1, "Churn_Prediction")  # Insert it at index 1
    merged_final_df = merged_final_df[cols]  # Reorder the DataFrame

    if output_file:

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_final_df.to_csv(output_file, index=False)

        print(f"✅ Predictions saved to {output_file}")
    else:
        print("⚠️ No output path provided. Please specify --output.")


### ------------------ Main Script ------------------ ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer churn using a pre-trained model.")

    required_args = parser.add_argument_group("Required arguments")
    required_args.add_argument("--activity", required=True, help="File of activity data CSV")
    required_args.add_argument("--customer", required=True, help="File of customer data CSV")
    required_args.add_argument("--complaints", required=True, help="File of complaints Excel")
    required_args.add_argument("--model_type", required=True, choices=["logistic", "lda", "adaboost"], help="Type of model to use for prediction")
    required_args.add_argument("--nlp", required=True, help="File trained NLP random forest classifier")
    required_args.add_argument("--model", required=True, help="File of trained model")
    required_args.add_argument("--scaler", help="File of trained scaler (not required for adaboost)")
    required_args.add_argument("--features", required=True, help="File with feature names")
    required_args.add_argument("--output", required=True, help="File to save predictions")

    args = parser.parse_args()

    # Check if scaler is required but missing
    if args.model_type in ["logistic", "lda"] and not args.scaler:
        print("Warning: --scaler is required for logistic and lda models.")
    else:
        # Run final_pred only if all required arguments are present
        final_pred(args.activity, args.customer, args.complaints, args.model_type, args.model, args.nlp, args.scaler, args.features, args.output)
