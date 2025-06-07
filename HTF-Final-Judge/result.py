import re
import socket
from urllib.parse import urlparse
import whois
from datetime import datetime
from googlesearch import search
import dns.resolver

http_https = r"https://|http://"
special_characters = """!@#$%^&*()_+}{[]|\:"';<>?,-1234567890"""
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"

def _url_domain(url):
    try:
        pr_url = urlparse(url)
    except Exception as e:
        return "Acces-restricted"
    return pr_url.netloc

def has_ip(domain_name):
    try:
        ip_address = socket.gethostbyname(domain_name)
    except Exception as e:
        ip_address = "Acces-restricted"
    return ip_address

def domain_age(domain_name):
    try:
        domain_info = whois.whois(domain_name)
        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date
        
        if creation_date:
            current_year = datetime.now().year
            domain_age = current_year - creation_date.year
            return domain_age
        else:
            return 0
    except Exception as e:
        return "Acces-restricted"

def number_of_subdomains(url):
    dot_count = len(re.findall("\.", url))
    return dot_count

def domain_registration_length(domain):
    try:
        domain_info = whois.whois(domain)
        if isinstance(domain_info.expiration_date, list):
            expiration_date = domain_info.expiration_date[0]
        else:
            expiration_date = domain_info.expiration_date
        if not expiration_date:
            return "Access-Restricted"
        today = datetime.now()
        registration_length = (expiration_date - today).days
        return registration_length
    except Exception as e:
        return "Acces-restricted"

def get_ip_count(domain):
    try:
        ips = socket.gethostbyname_ex(domain)[2]
        return len(ips)
    except socket.gaierror as e:
        return "Acces-restricted"

def get_ssl_update_age(domain):
    try:
        whois_response = whois.whois(domain)
        cert_data = whois_response["updated_date"]
        if isinstance(cert_data, list):
            cert_data = cert_data[0]
        given_date_time_str = str(cert_data)
        given_date_time = datetime.strptime(given_date_time_str, "%Y-%m-%d %H:%M:%S")
        current_date_time = datetime.utcnow()
        age = current_date_time - given_date_time
        return age.days
    except Exception as e:
        return "Acces-restricted"

def number_of_smtp_servers(domain):
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        if len(mx_records) > 0:
            return len(mx_records)
        else:
            return 0
    except Exception as e:
        return "Acces-restricted"

#domain_age("https://google.com")
print(domain_age("https://www.bietdvg.edu/"))