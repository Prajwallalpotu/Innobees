import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ResumeProjectRecommender:
    def __init__(self):
        # Pre-defined tech domains and their related keywords
        self.domains = {
            'Frontend': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'ui', 'ux', 'responsive', 'bootstrap', 'tailwind'],
            'Backend': ['python', 'java', 'node', 'express', 'django', 'flask', 'api', 'rest', 'graphql', 'php', 'laravel', 'spring'],
            'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'nosql', 'redis', 'oracle', 'database design'],
            'DevOps': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'linux'],
            'Mobile': ['android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'mobile app'],
            'AI/ML': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision', 'data science', 'neural network'],
            'Blockchain': ['blockchain', 'ethereum', 'solidity', 'smart contract', 'web3', 'cryptocurrency'],
            'Security': ['security', 'encryption', 'penetration testing', 'ethical hacking', 'firewall', 'cybersecurity']
        }
        
        # Sample project ideas with required skills
        self.projects = [
            {
                'name': 'E-commerce Platform',
                'description': 'Build a full-stack e-commerce platform with product listings, user authentication, shopping cart, and payment integration.',
                'problem_solved': 'Helps small businesses establish an online presence and sell products directly to customers.',
                'skills_required': {
                    'Frontend': 0.4,
                    'Backend': 0.3,
                    'Database': 0.3
                }
            },
            {
                'name': 'Healthcare Appointment System',
                'description': 'Create a system for patients to book appointments with healthcare providers, including reminders and medical history tracking.',
                'problem_solved': 'Streamlines the healthcare appointment process, reducing wait times and improving patient experience.',
                'skills_required': {
                    'Frontend': 0.3,
                    'Backend': 0.4,
                    'Database': 0.3
                }
            },
            {
                'name': 'Predictive Maintenance System',
                'description': 'Develop a system that predicts when industrial equipment will need maintenance using sensor data and machine learning.',
                'problem_solved': 'Reduces unexpected downtime in manufacturing and saves costs on emergency repairs.',
                'skills_required': {
                    'AI/ML': 0.6,
                    'Backend': 0.2,
                    'Frontend': 0.2
                }
            },
            {
                'name': 'Inventory Management System',
                'description': 'Build a system to track inventory levels, orders, sales, and deliveries for businesses.',
                'problem_solved': 'Helps businesses optimize stock levels and reduce losses from overstocking or stockouts.',
                'skills_required': {
                    'Backend': 0.4,
                    'Database': 0.3,
                    'Frontend': 0.3
                }
            },
            {
                'name': 'Content Recommendation Engine',
                'description': 'Create a system that recommends content to users based on their preferences and behavior.',
                'problem_solved': 'Increases user engagement on content platforms by providing personalized recommendations.',
                'skills_required': {
                    'AI/ML': 0.5,
                    'Backend': 0.3,
                    'Frontend': 0.2
                }
            },
            {
                'name': 'Supply Chain Tracking System',
                'description': 'Develop a blockchain-based system to track products through the supply chain from manufacturer to consumer.',
                'problem_solved': 'Increases transparency in supply chains and helps verify authenticity of products.',
                'skills_required': {
                    'Blockchain': 0.6,
                    'Backend': 0.2,
                    'Frontend': 0.2
                }
            },
            {
                'name': 'Automated Deployment Pipeline',
                'description': 'Build a CI/CD pipeline that automates testing, building, and deployment of applications.',
                'problem_solved': 'Streamlines the development process and reduces time-to-market for software products.',
                'skills_required': {
                    'DevOps': 0.7,
                    'Backend': 0.3
                }
            },
            {
                'name': 'Secure File Sharing Platform',
                'description': 'Create a platform for secure file sharing with end-to-end encryption and access controls.',
                'problem_solved': 'Enables businesses to share sensitive documents securely with clients and partners.',
                'skills_required': {
                    'Security': 0.5,
                    'Backend': 0.3,
                    'Frontend': 0.2
                }
            },
            {
                'name': 'Real-time Chat Application',
                'description': 'Build a real-time chat application with features like group chats, file sharing, and voice messages.',
                'problem_solved': 'Improves team communication and collaboration, especially for remote teams.',
                'skills_required': {
                    'Frontend': 0.4,
                    'Backend': 0.4,
                    'Database': 0.2
                }
            },
            {
                'name': 'Smart Home Automation System',
                'description': 'Develop a system to control and automate home devices like lights, thermostats, and security systems.',
                'problem_solved': 'Increases home energy efficiency and improves convenience and security for homeowners.',
                'skills_required': {
                    'Backend': 0.3,
                    'Frontend': 0.3,
                    'Mobile': 0.4
                }
            }
        ]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def extract_skills_from_resume(self, resume_text):
        """Extract skills from resume text and map to domains"""
        resume_text = resume_text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domains.items():
            domain_score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', resume_text))
                if count > 0:
                    domain_score += count
            
            domain_scores[domain] = domain_score
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            for domain in domain_scores:
                domain_scores[domain] = domain_scores[domain] / total_score
        
        return domain_scores

    def calculate_project_match(self, user_domain_scores, project):
        """Calculate match score between user skills and project requirements"""
        match_score = 0
        for domain, required_score in project['skills_required'].items():
            user_score = user_domain_scores.get(domain, 0)
            match_score += min(user_score, required_score)
        
        return match_score

    def recommend_projects(self, resume_text, num_recommendations=3):
        """Recommend projects based on resume"""
        user_domain_scores = self.extract_skills_from_resume(resume_text)
        
        # Calculate match scores for each project
        project_matches = []
        for project in self.projects:
            match_score = self.calculate_project_match(user_domain_scores, project)
            project_matches.append((project, match_score))
        
        # Sort projects by match score (descending)
        project_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return [project for project, score in project_matches[:num_recommendations]]

    def recommend_team_projects(self, resumes, num_recommendations=3):
        """Recommend projects for a team based on their resumes"""
        team_size = len(resumes)
        team_domain_scores = {}
        individual_domain_scores = []
        
        # Calculate domain scores for each team member
        for resume in resumes:
            member_scores = self.extract_skills_from_resume(resume)
            individual_domain_scores.append(member_scores)
            
            # Aggregate team scores
            for domain, score in member_scores.items():
                if domain in team_domain_scores:
                    team_domain_scores[domain] += score
                else:
                    team_domain_scores[domain] = score
        
        # Normalize team scores
        for domain in team_domain_scores:
            team_domain_scores[domain] /= team_size
        
        # Calculate match scores for each project
        project_matches = []
        for project in self.projects:
            match_score = self.calculate_project_match(team_domain_scores, project)
            project_matches.append((project, match_score))
        
        # Sort projects by match score (descending)
        project_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N project recommendations
        recommended_projects = []
        for project, score in project_matches[:num_recommendations]:
            # For each recommended project, assign roles to team members
            member_assignments = self.assign_team_roles(project, individual_domain_scores)
            
            project_with_assignments = project.copy()
            project_with_assignments['team_assignments'] = member_assignments
            recommended_projects.append(project_with_assignments)
        
        return recommended_projects
    
    def assign_team_roles(self, project, individual_domain_scores):
        """Assign team members to roles based on their skills and project requirements"""
        required_domains = list(project['skills_required'].keys())
        team_size = len(individual_domain_scores)
        assignments = [[] for _ in range(team_size)]
        
        # Sort domains by required score (descending)
        sorted_domains = sorted(required_domains, 
                               key=lambda d: project['skills_required'][d], 
                               reverse=True)
        
        # For each domain, find the best team member
        for domain in sorted_domains:
            # Get scores for this domain from all team members
            domain_scores = [scores.get(domain, 0) for scores in individual_domain_scores]
            
            # Find the team member with the highest score for this domain
            best_member = np.argmax(domain_scores)
            
            # Assign this domain to the best member
            assignments[best_member].append(domain)
        
        return assignments

    def add_custom_project(self, project_data):
        """Add a custom project to the recommendation pool"""
        self.projects.append(project_data)
        
    def format_recommendations(self, recommendations, is_team=False):
        """Format project recommendations for display"""
        result = []
        
        for i, project in enumerate(recommendations):
            formatted_project = {
                'name': project['name'],
                'description': project['description'],
                'problem_solved': project['problem_solved'],
                'required_skills': {k: f"{v*100:.0f}%" for k, v in project['skills_required'].items()}
            }
            
            if is_team:
                team_roles = []
                for member_idx, domains in enumerate(project['team_assignments']):
                    if domains:
                        team_roles.append({
                            'member': f"Team Member {member_idx+1}",
                            'responsibilities': domains
                        })
                formatted_project['team_assignments'] = team_roles
                
            result.append(formatted_project)
            
        return result


# Example usage
if __name__ == "__main__":
    recommender = ResumeProjectRecommender()
    
    # Example for individual
    sample_resume = """
    John Doe
    Software Engineer
    
    Skills:
    - Proficient in Python, JavaScript, and React
    - Experience with Django and Flask frameworks
    - Database design and optimization (MySQL, PostgreSQL)
    - RESTful API development
    
    Experience:
    Software Engineer at Tech Company (2020-2023)
    - Developed and maintained frontend applications using React
    - Built backend services with Python and Flask
    - Designed database schemas for new features
    
    Education:
    Bachelor of Science in Computer Science
    """
    
    individual_recommendations = recommender.recommend_projects(sample_resume)
    print("Individual Recommendations:")
    for rec in recommender.format_recommendations(individual_recommendations):
        print(f"- {rec['name']}: {rec['description']}")
        print(f"  Problem solved: {rec['problem_solved']}")
        print(f"  Required skills: {rec['required_skills']}")
        print()
    
    # Example for team
    team_resumes = [
        """
        Alice Smith
        Frontend Developer
        
        Skills:
        - Expert in HTML, CSS, JavaScript
        - React, Angular, Vue.js
        - Responsive design
        - UI/UX principles
        
        Experience:
        Frontend Developer at Web Studio (2018-2023)
        - Created responsive web applications
        - Implemented UI/UX designs
        - Optimized website performance
        """,
        
        """
        Bob Johnson
        Data Scientist
        
        Skills:
        - Machine Learning (TensorFlow, PyTorch)
        - Data Analysis (Pandas, NumPy)
        - Python, R
        - NLP and Computer Vision
        
        Experience:
        Data Scientist at AI Labs (2019-2023)
        - Developed recommendation systems
        - Implemented computer vision solutions
        - Created predictive models
        """,
        
        """
        Charlie Brown
        Backend Developer
        
        Skills:
        - Java, Python, Node.js
        - SQL and NoSQL databases
        - RESTful API design
        - Microservices architecture
        
        Experience:
        Backend Engineer at Server Solutions (2017-2023)
        - Designed and implemented APIs
        - Optimized database performance
        - Developed microservices
        """
    ]
    
    team_recommendations = recommender.recommend_team_projects(team_resumes)
    print("\nTeam Recommendations:")
    for rec in recommender.format_recommendations(team_recommendations, is_team=True):
        print(f"- {rec['name']}: {rec['description']}")
        print(f"  Problem solved: {rec['problem_solved']}")
        print(f"  Required skills: {rec['required_skills']}")
        print("  Team assignments:")
        for role in rec['team_assignments']:
            print(f"    * {role['member']}: {', '.join(role['responsibilities'])}")
        print()