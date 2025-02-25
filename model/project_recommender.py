def recommend_projects(skills, tech_stack):
    projects = {
        "AI/ML": ["AI Chatbot", "Image Recognition App", "Sentiment Analysis Tool"],
        "Web Development": ["E-commerce Website", "Portfolio Builder", "Social Media Dashboard"],
        "Data Science": ["Stock Price Predictor", "Customer Segmentation Model", "Recommendation System"],
    }

    roles = {
        "Frontend": "UI/UX, React, HTML, CSS, JavaScript",
        "Backend": "Flask, Node.js, Databases",
        "AI/ML": "TensorFlow, Scikit-learn, Deep Learning",
    }

    domain = "AI/ML" if "ml" in tech_stack or "ai" in tech_stack else "Web Development"

    project_suggestions = projects.get(domain, ["Custom Project Idea"])
    assigned_roles = {role: details for role, details in roles.items() if any(skill.lower() in details.lower() for skill in tech_stack)}

    return project_suggestions, assigned_roles
