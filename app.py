from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import pdfplumber
import os
import torch

app = Flask(__name__)

# Load NLP model for skill extraction
nlp = spacy.load("en_core_web_sm")

# Load Flan-T5 XXL model
tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")
model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")

def extract_skills(text):
    """Extract skills from resume text using spaCy."""
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE"]]
    return list(set(skills))  # Remove duplicates

def recommend_projects(skills):
    """Use Flan-T5 XXL to generate project ideas based on skills."""
    if not skills:
        return ["No relevant skills found in the resume."]
    
    prompt = f"I have skills in {', '.join(skills)}. Suggest three real-world project ideas related to my skills."

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cpu")  # Use "cuda" if GPU available
    
    # Generate response from the model
    outputs = model.generate(**inputs, max_length=200)
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Split into multiple project ideas
    return generated_text.split("\n")

def assign_roles(skills):
    """Assign roles to team members based on individual strengths."""
    roles = {
        "Frontend": "UI/UX, React, HTML, CSS, JavaScript",
        "Backend": "Flask, Node.js, Databases",
        "AI/ML": "TensorFlow, Scikit-learn, Deep Learning",
    }

    assigned_roles = {}
    for role, skills_required in roles.items():
        for skill in skills:
            if skill.lower() in skills_required.lower():
                assigned_roles[role] = skills_required
                break

    return assigned_roles

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        file = request.files["resume"]
        if file.filename == "":
            return render_template("index.html", error="No file uploaded.")
        
        # Save file temporarily
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Extract text from PDF
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        skills = extract_skills(text)
        projects = recommend_projects(skills)
        roles = assign_roles(skills)

        return render_template("results.html", skills=skills, projects=projects, roles=roles)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)