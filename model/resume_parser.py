import spacy
import pdfplumber

nlp = spacy.load("en_core_web_sm")

def extract_skills(resume_path):
    with pdfplumber.open(resume_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ == "NOUN" or token.ent_type_ == "ORG"]

    tech_stack = [word for word in skills if word.lower() in ["python", "java", "react", "flask", "tensorflow", "ml", "ai"]]

    return skills, tech_stack
