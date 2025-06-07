import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib
import result as ress

model_path = "best_rf_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

def preprocess_url(domain):
    if "." not in domain:
        raise ValueError("Invalid URL format")
    if not domain.startswith("http://") and not domain.startswith("https://"):
        domain = "https://" + domain
    if domain.startswith("https://www."):
        domain = "https://" + domain[12:]
    elif domain.startswith("http://www."):
        domain = "http://" + domain[11:]
    return domain

def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        raise ConnectionError(f"Connection Error: {e}")

def extract_features(domain):
    features = fex.data_set_list_creation(domain)
    if features is None or not isinstance(features, list):
        raise ValueError("Feature extraction failed or returned invalid data")
    return features

def predict_phishing(features):
    if not features:
        raise ValueError("Features are empty or None")
    features = np.array([features])
    prediction = model.predict(features)
    if prediction is None:
        raise ValueError("Prediction failed or returned None")
    
    return prediction[0] == 1

def process_url_input(domain_input):
    try:
        cleaned_domain = preprocess_url(domain_input)
        print(f"Cleaned domain: {cleaned_domain}")
        fetch_url_content(cleaned_domain)
        print(f"URL fetched successfully: {cleaned_domain}")
        features = extract_features(cleaned_domain)
        print(f"Extracted features: {features}")
        print(len(features));
        is_phishing = predict_phishing(features)
        result_text = "The URL is predicted as a phishing domain🔴." if is_phishing else "The URL is predicted as a legitimate domain🟢."
        
        additional_info = {
            "domain": str(ress._url_domain(cleaned_domain)),
            "ip": str(ress.has_ip(ress._url_domain(cleaned_domain))), 
            "Domain Age": str(ress.domain_age(ress._url_domain(cleaned_domain))),
            "num_sub_domains": str(ress.number_of_subdomains(preprocess_url(cleaned_domain))), 
            "domain_reg_length": str(ress.domain_registration_length(ress._url_domain(cleaned_domain))),  
            "ip_counts": str(ress.get_ip_count(ress._url_domain(cleaned_domain))), 
            "ssl_update_age(In Days)": str(ress.get_ssl_update_age(ress._url_domain(cleaned_domain))),  
            "num_smtp_servers": str(ress.number_of_smtp_servers(ress._url_domain(cleaned_domain).removeprefix("www.")))  
        }
        
        return {
            "result_text": result_text,
            "additional_info": additional_info
        }

    except (ConnectionError) as e:
        return {
            "result_text": f"Error: {e}",
            "additional_info": {}
        }