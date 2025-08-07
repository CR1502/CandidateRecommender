"""
Script to generate sample resume files for testing.
"""

from pathlib import Path
import random

# Sample resume templates
RESUME_TEMPLATES = [
    {
        "name": "Alice Johnson",
        "title": "Senior Python Developer",
        "content": """
Alice Johnson
Senior Python Developer
Email: alice.johnson@email.com | Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Experienced Python developer with 7+ years building scalable web applications and data pipelines.
Expert in Django, FastAPI, and machine learning frameworks including TensorFlow and PyTorch.
Strong background in microservices architecture and cloud deployment.

TECHNICAL SKILLS
Languages: Python, JavaScript, SQL, Bash
Frameworks: Django, FastAPI, Flask, React
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
ML/AI: TensorFlow, PyTorch, scikit-learn, pandas, numpy

PROFESSIONAL EXPERIENCE
Senior Python Developer | TechCorp Inc. | 2020-Present
- Architected and developed RESTful APIs serving 1M+ daily requests
- Implemented machine learning pipelines for recommendation system
- Led team of 5 developers in microservices migration
- Reduced API response time by 40% through optimization

Python Developer | DataSoft Solutions | 2017-2020
- Built data processing pipelines handling 10TB+ daily
- Developed Django web applications for enterprise clients
- Integrated third-party APIs and payment gateways
- Mentored junior developers and conducted code reviews

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2016
"""
    },
    {
        "name": "Bob Smith",
        "title": "Full Stack Developer",
        "content": """
Bob Smith
Full Stack Developer
Email: bob.smith@email.com | LinkedIn: linkedin.com/in/bobsmith

PROFILE
Versatile full stack developer with 5 years of experience in web development.
Proficient in Python backend development and modern JavaScript frameworks.
Passionate about clean code and agile methodologies.

SKILLS
• Backend: Python, Node.js, Express, Django
• Frontend: React, Vue.js, HTML5, CSS3, JavaScript
• Database: MySQL, PostgreSQL, MongoDB
• Tools: Git, Docker, Jenkins, JIRA

WORK EXPERIENCE
Full Stack Developer | WebDev Co. | 2021-Present
- Develop and maintain Python/Django backend services
- Create responsive React frontend applications
- Implement CI/CD pipelines using Jenkins and Docker
- Collaborate with design team on UX improvements

Junior Developer | StartupXYZ | 2019-2021
- Built RESTful APIs using Python and Flask
- Developed interactive dashboards with Vue.js
- Wrote unit tests and integration tests
- Participated in agile sprint planning

EDUCATION
Bachelor of Science in Software Engineering
State University | 2018

CERTIFICATIONS
AWS Certified Developer - Associate
Python Institute Certified Professional
"""
    },
    {
        "name": "Carol Martinez",
        "title": "Machine Learning Engineer",
        "content": """
Carol Martinez
Machine Learning Engineer
carol.martinez@email.com | GitHub: github.com/carolmartinez

SUMMARY
Machine Learning Engineer with 6 years of experience developing and deploying ML models.
Specialized in NLP, computer vision, and deep learning with Python.
Strong mathematical background and experience with production ML systems.

TECHNICAL EXPERTISE
Programming: Python, R, SQL, C++
ML Frameworks: TensorFlow, PyTorch, Keras, scikit-learn
Deep Learning: CNNs, RNNs, Transformers, GANs
Data Tools: pandas, numpy, Spark, Hadoop
Cloud ML: AWS SageMaker, Google Cloud AI Platform

PROFESSIONAL EXPERIENCE
Senior ML Engineer | AI Innovations Inc. | 2021-Present
• Developed state-of-the-art NLP models using transformers
• Built computer vision system for quality inspection (95% accuracy)
• Deployed ML models to production using Docker and Kubernetes
• Reduced model inference time by 60% through optimization

Machine Learning Engineer | DataTech Corp | 2019-2021
• Implemented recommendation system using collaborative filtering
• Created data pipelines for feature engineering
• Developed A/B testing framework for model evaluation
• Published research paper on neural network optimization

Data Scientist | Analytics Pro | 2018-2019
• Built predictive models using Python and scikit-learn
• Performed statistical analysis and data visualization
• Created automated reporting dashboards

EDUCATION
Master of Science in Machine Learning
Tech Institute | 2018

Bachelor of Science in Mathematics
University of Sciences | 2016

PUBLICATIONS
"Optimizing Deep Neural Networks for Edge Deployment" - ML Conference 2022
"""
    },
    {
        "name": "David Chen",
        "title": "DevOps Engineer",
        "content": """
David Chen
DevOps Engineer
david.chen@email.com | Phone: (555) 987-6543

PROFESSIONAL SUMMARY
DevOps Engineer with 5+ years automating infrastructure and deployment pipelines.
Expert in Python scripting, cloud platforms, and container orchestration.
Strong focus on reliability, monitoring, and performance optimization.

TECHNICAL SKILLS
Languages: Python, Bash, Go, JavaScript
Infrastructure: Terraform, Ansible, CloudFormation
Containers: Docker, Kubernetes, OpenShift
CI/CD: Jenkins, GitLab CI, GitHub Actions
Cloud: AWS, Azure, GCP
Monitoring: Prometheus, Grafana, ELK Stack

EXPERIENCE
Senior DevOps Engineer | CloudScale Systems | 2021-Present
- Architected Kubernetes clusters serving 100+ microservices
- Automated infrastructure provisioning with Terraform and Python
- Implemented GitOps workflow reducing deployment time by 70%
- Built monitoring and alerting system with Prometheus/Grafana

DevOps Engineer | TechOps Inc. | 2019-2021
- Developed Python automation scripts for deployment
- Managed AWS infrastructure for production applications
- Created CI/CD pipelines using Jenkins and Docker
- Improved system reliability from 99.5% to 99.99% uptime

Junior DevOps Engineer | StartupCo | 2018-2019
- Assisted in cloud migration to AWS
- Wrote Ansible playbooks for configuration management
- Monitored application performance and resolved issues

EDUCATION
Bachelor of Science in Computer Engineering
Engineering University | 2017

CERTIFICATIONS
AWS Certified Solutions Architect
Certified Kubernetes Administrator (CKA)
"""
    },
    {
        "name": "Emma Wilson",
        "title": "Data Engineer",
        "content": """
Emma Wilson
Data Engineer
emma.wilson@email.com | Portfolio: emmawilson.dev

ABOUT ME
Data Engineer with 4 years of experience building scalable data pipelines.
Proficient in Python, SQL, and big data technologies.
Experienced in ETL development and data warehouse design.

SKILLS & TECHNOLOGIES
• Programming: Python, SQL, Scala, Java
• Big Data: Apache Spark, Hadoop, Kafka, Airflow
• Databases: PostgreSQL, MySQL, Redshift, BigQuery
• Cloud: AWS (S3, EMR, Glue), GCP (Dataflow, BigQuery)
• Tools: Git, Docker, dbt, Tableau

WORK EXPERIENCE
Data Engineer | DataFlow Corp | 2021-Present
- Design and implement ETL pipelines processing 5TB+ daily
- Develop Python scripts for data transformation and validation
- Build real-time streaming pipelines with Kafka and Spark
- Optimize query performance reducing runtime by 50%

Junior Data Engineer | Analytics Hub | 2020-2021
- Created data pipelines using Apache Airflow and Python
- Maintained data warehouse in AWS Redshift
- Developed data quality monitoring frameworks
- Collaborated with data scientists on feature engineering

Data Analyst | Business Insights Co. | 2019-2020
- Wrote complex SQL queries for business reporting
- Built automated dashboards using Tableau
- Performed data analysis using Python and pandas

EDUCATION
Master of Science in Data Science
Data University | 2019

Bachelor of Science in Statistics
State College | 2017

PROJECTS
• Real-time fraud detection pipeline using Spark Streaming
• Data lake architecture implementation on AWS
"""
    }
]

# Non-matching resumes for testing
NON_MATCHING_TEMPLATES = [
    {
        "name": "Frank Miller",
        "title": "Marketing Manager",
        "content": """
Frank Miller
Marketing Manager
frank.miller@email.com

SUMMARY
Marketing professional with 8 years of experience in digital marketing and brand management.
Expert in campaign strategy, social media marketing, and content creation.

SKILLS
• Digital Marketing: SEO, SEM, PPC, Email Marketing
• Social Media: Facebook, Instagram, LinkedIn, Twitter
• Analytics: Google Analytics, Adobe Analytics
• Tools: HubSpot, Salesforce, Mailchimp

EXPERIENCE
Marketing Manager | BrandCo | 2020-Present
- Lead marketing team of 6 professionals
- Develop and execute marketing strategies
- Manage $2M annual marketing budget
- Increased brand awareness by 40%

EDUCATION
MBA in Marketing
Business School | 2015
"""
    },
    {
        "name": "Grace Lee",
        "title": "Graphic Designer",
        "content": """
Grace Lee
Graphic Designer
grace.lee@email.com

PROFILE
Creative graphic designer with 5 years of experience in visual design and branding.
Proficient in Adobe Creative Suite and modern design principles.

SKILLS
• Software: Photoshop, Illustrator, InDesign, Figma
• Design: Typography, Color Theory, Layout Design
• Web: HTML, CSS, Basic JavaScript

EXPERIENCE
Senior Graphic Designer | Design Studio | 2021-Present
- Create visual designs for print and digital media
- Develop brand identity systems
- Collaborate with marketing teams
- Mentor junior designers

EDUCATION
Bachelor of Fine Arts in Graphic Design
Art Institute | 2018
"""
    }
]


def generate_sample_resumes():
    """Generate sample resume files for testing."""
    # Create directory
    sample_dir = Path("data/sample_resumes")
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample resumes...")

    # Generate matching resumes
    for i, template in enumerate(RESUME_TEMPLATES, 1):
        filename = f"{template['name'].replace(' ', '_').lower()}_resume.txt"
        filepath = sample_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template['content'])

        print(f"✓ Created: {filename}")

    # Generate non-matching resumes
    for template in NON_MATCHING_TEMPLATES:
        filename = f"{template['name'].replace(' ', '_').lower()}_resume.txt"
        filepath = sample_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template['content'])

        print(f"✓ Created: {filename}")

    print(f"\nGenerated {len(RESUME_TEMPLATES) + len(NON_MATCHING_TEMPLATES)} sample resumes in {sample_dir}")
    print("\nSample resumes include:")
    print("- 5 Python/Tech professionals (should match well)")
    print("- 2 Non-tech professionals (should have low match)")


if __name__ == "__main__":
    generate_sample_resumes()