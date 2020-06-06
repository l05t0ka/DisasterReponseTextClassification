import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
#import nltk
from sklearn.base import BaseEstimator, TransformerMixin

def tokenize(text):
    """tokenize function processes message text data - lowercase, no punctuation, lemmatization, URLs
    INPUT: text (str) - text of message
    OUTPUT: cleaned_tokens (list) - List of tokens"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_found = re.findall(url_regex, text)
    for url in urls_found:
        text = text.replace(url, 'URL')
    email_regex = '\w+@\w+\.\w+'
    emails_found = re.findall(email_regex, text)
    for email in emails_found:
        text = text.replace(email, 'EMAIL')
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
    return cleaned_tokens

class MessagePersonalPronounsUsed(BaseEstimator, TransformerMixin):
    """Transformer: Checks if personal pronouns were used in a message and outputs their number of occurence"""
    
    def calc_personal_pronouns(self, text):
        text = re.sub(r"[^a-zA-Z]"," ", text)
        PRP_list = []
        tagged_list = pos_tag(word_tokenize(text))
        for word, pos in tagged_list:
            if pos == "PRP":
                PRP_list.append(word)
        return len(PRP_list)
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.calc_personal_pronouns)
        return pd.DataFrame(X_tagged)
    


class MessagePossesivePronounsUsed(BaseEstimator, TransformerMixin):
    """Transformer: Checks if possesive pronouns were used in a message and outputs their number of occurence"""
    
    def calc_possesive_pronouns(self, text):
        text = re.sub(r"[^a-zA-Z]"," ", text)
        PRPS_list = []
        tagged_list = pos_tag(word_tokenize(text))
        for word, pos in tagged_list:
            if pos == "PRP$":
                PRPS_list.append(word)
        return len(PRPS_list)
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.calc_possesive_pronouns)
        return pd.DataFrame(X_tagged)

class MessageNounsUsed(BaseEstimator, TransformerMixin):
    """Transformer: Checks if common nouns were used in a message and outputs their number of occurence"""
    
    def calc_common_nouns(self, text):
        text = re.sub(r"[^a-zA-Z]"," ", text)
        NN_list = []
        tagged_list = pos_tag(word_tokenize(text))
        for word, pos in tagged_list:
            if pos == "NN" or pos=="NNS":
                NN_list.append(word)
        return len(NN_list)
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.calc_common_nouns)
        return pd.DataFrame(X_tagged)

class MessageProperNounsUsed(BaseEstimator, TransformerMixin):
    """Transformer: Checks if possesive pronouns were used in a message and outputs their number of occurence"""
    
    def calc_proper_nouns(self, text):
        text = re.sub(r"[^a-zA-Z]"," ", text)
        NNP_list = []
        tagged_list = pos_tag(word_tokenize(text))
        for word, pos in tagged_list:
            if pos=="NNP":
                NNP_list.append(word)
        return len(NNP_list)
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.calc_proper_nouns)
        return pd.DataFrame(X_tagged)

class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    """Transformer: Extracts length of a message"""
    
    def extract_length(self, text):
        text = re.sub(r"[^a-zA-Z0-9]"," ", text)
        return len(text.split())
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.extract_length)
        return pd.DataFrame(X_len)

class MessageAvgWordExtractor(BaseEstimator, TransformerMixin):
    """Transformer: Extracts average word length of a message"""
    
    def extract_avg_word_length(self, text): 
        text = re.sub(r"[^a-zA-Z0-9]"," ", text)
        text_splitted = text.split()
        try:
            output = (sum(len(word) for word in text_splitted)/len(text_splitted))
        except ZeroDivisionError:
            output = 0
        return output
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.extract_avg_word_length)
        return pd.DataFrame(X_len)

class MessageNumericsExtractor(BaseEstimator, TransformerMixin):
    """Transformer: Extracts number of numbers in a message"""
    
    def extract_number_numerics(self, text): 
        text = re.sub(r"[^0-9]"," ", text)
        text_splitted = text.split()
        return len(text_splitted)
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.extract_number_numerics)
        return pd.DataFrame(X_len)

class MessageUppercaseExtractor(BaseEstimator, TransformerMixin):
    """Transformer: Extracts number of words written in uppercase"""
    def extract_uppercase(self, text): 
        text = re.sub(r"[^a-zA-Z0-9]"," ", text)
        return len([word for word in text.split() if word.isupper()])
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.extract_uppercase)
        return pd.DataFrame(X_len)

class UrlsEmailsExtractor(BaseEstimator, TransformerMixin):
    """Transformer: Extracts number of urls and emails a message contains"""
    def extract_number_urls_and_emails(self, text): 
        regex = '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|\w+@\w+\.\w+)'        
        return len(re.findall(regex, text))
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.extract_number_urls_and_emails)
        return pd.DataFrame(X_len)



    
    
