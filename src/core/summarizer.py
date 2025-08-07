"""
AI-powered summarization for candidate fit explanations.
"""

from typing import Dict, Any, List, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from loguru import logger


class CandidateSummarizer:
    """Generate AI summaries explaining why candidates are good fits."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the summarizer with a local T5 model.
        
        Args:
            model_name: Name of the T5 model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 512
        self.summary_length = 100
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading summarization model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Summarization model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            # Fall back to template-based summaries
            self.model = None
            self.tokenizer = None
    
    def generate_fit_summary(
        self,
        job_description: str,
        resume_text: str,
        similarity_score: float,
        matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate a summary explaining why the candidate is a good fit.
        
        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score (0-1)
            matching_skills: Optional list of matching skills
            
        Returns:
            Generated summary text
        """
        # If model is available, use AI generation
        if self.model and self.tokenizer:
            try:
                return self._generate_ai_summary(
                    job_description, 
                    resume_text, 
                    similarity_score,
                    matching_skills
                )
            except Exception as e:
                logger.warning(f"AI summary generation failed: {e}")
                # Fall back to template
        
        # Use template-based summary as fallback
        return self._generate_template_summary(
            job_description,
            resume_text,
            similarity_score,
            matching_skills
        )
    
    def _generate_ai_summary(
        self,
        job_description: str,
        resume_text: str,
        similarity_score: float,
        matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate AI-powered summary using T5 model.
        
        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score
            matching_skills: Optional matching skills
            
        Returns:
            AI-generated summary
        """
        # Truncate texts to fit model limits
        job_desc_truncated = job_description[:500]
        resume_truncated = resume_text[:500]
        
        # Create prompt
        prompt = f"""
        Task: Explain why this candidate is a good fit for the job in 2-3 sentences.
        
        Job requirements: {job_desc_truncated}
        
        Candidate background: {resume_truncated}
        
        Summary:
        """
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.summary_length,
                min_length=30,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_beams=2
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up and enhance summary
        if matching_skills and len(matching_skills) > 0:
            skills_text = ", ".join(matching_skills[:3])
            summary += f" Key matching skills include {skills_text}."
        
        return summary.strip()
    
    def _generate_template_summary(
        self,
        job_description: str,
        resume_text: str,
        similarity_score: float,
        matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate diverse, contextual summaries as fallback.
        
        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score
            matching_skills: Optional matching skills
            
        Returns:
            Contextually rich summary
        """
        from .text_cleaner import TextCleaner
        import random
        import re
        
        cleaner = TextCleaner()
        
        # Extract skills if not provided
        if not matching_skills:
            job_skills = set(cleaner.extract_key_skills(job_description))
            resume_skills = set(cleaner.extract_key_skills(resume_text))
            matching_skills = list(job_skills & resume_skills)
        
        # Analyze resume for experience indicators
        years_pattern = r'(\d+)\+?\s*years?'
        years_matches = re.findall(years_pattern, resume_text.lower())
        max_years = max([int(y) for y in years_matches], default=0)
        
        # Check for leadership/senior indicators
        leadership_keywords = ['lead', 'senior', 'manager', 'head', 'director', 'principal', 'architect']
        has_leadership = any(keyword in resume_text.lower() for keyword in leadership_keywords)
        
        # Check for education level
        has_masters = 'master' in resume_text.lower() or 'mba' in resume_text.lower()
        has_phd = 'phd' in resume_text.lower() or 'ph.d' in resume_text.lower() or 'doctorate' in resume_text.lower()
        
        # Analyze job requirements
        job_lower = job_description.lower()
        needs_ml = any(term in job_lower for term in ['machine learning', 'ml', 'ai', 'deep learning', 'neural'])
        needs_cloud = any(term in job_lower for term in ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker'])
        needs_backend = any(term in job_lower for term in ['backend', 'api', 'microservice', 'database', 'server'])
        needs_frontend = any(term in job_lower for term in ['frontend', 'react', 'angular', 'vue', 'ui/ux'])
        
        # Build diverse summaries based on score and context
        if similarity_score >= 0.8:
            templates = [
                f"Exceptional match with {max_years}+ years of directly relevant experience. {self._get_strength_statement(matching_skills, needs_ml, needs_cloud, has_leadership)}",
                f"This candidate's profile aligns remarkably well with your requirements. {self._get_experience_highlight(max_years, has_leadership, has_masters, has_phd)} {self._get_skills_statement(matching_skills)}",
                f"Outstanding candidate bringing proven expertise in {', '.join(matching_skills[:3])} along with {self._get_experience_descriptor(max_years, has_leadership)}. Their background demonstrates exactly the kind of hands-on experience you're seeking.",
                f"Near-perfect alignment with the role requirements. {self._get_unique_value_prop(matching_skills, max_years, has_leadership, needs_ml, needs_cloud)}",
            ]
        elif similarity_score >= 0.6:
            templates = [
                f"Strong candidate with solid experience in {', '.join(matching_skills[:3])}. {self._get_growth_statement(max_years, has_leadership)} Would likely excel in this position.",
                f"Well-qualified professional whose background in {self._get_skill_area(matching_skills, needs_ml, needs_backend)} aligns nicely with your needs. {self._get_potential_statement(has_masters, max_years)}",
                f"This candidate brings valuable expertise, particularly in {', '.join(matching_skills[:2])}. {self._get_fit_assessment(similarity_score, max_years, has_leadership)}",
                f"Compelling background with {max_years if max_years > 0 else 'relevant'} years of experience. {self._get_transferable_skills(matching_skills, needs_ml, needs_cloud)}",
            ]
        elif similarity_score >= 0.4:
            templates = [
                f"Solid foundational skills in {', '.join(matching_skills[:2]) if matching_skills else 'relevant areas'}. {self._get_development_potential(max_years, has_masters)}",
                f"This candidate shows promise with experience in {self._get_relevant_areas(matching_skills, resume_text)}. Could be a good fit with some additional training.",
                f"Interesting profile with transferable skills. {self._get_adjacent_experience(matching_skills, max_years)} Worth considering for the role.",
                f"Demonstrates competency in several key areas. {self._get_growth_trajectory(has_leadership, max_years, matching_skills)}",
            ]
        else:
            templates = [
                f"While primarily experienced in adjacent areas, this candidate shows {self._get_potential_indicator(matching_skills, has_masters)}",
                f"Alternative background that could bring fresh perspective. {self._get_transferable_value(matching_skills, max_years)}",
                f"Foundational skills present with room for growth. {self._get_learning_potential(has_masters, has_phd, max_years)}",
                f"Cross-functional experience that may translate well. {self._get_unique_angle(matching_skills, resume_text)}",
            ]
        
        # Select random template for diversity
        summary = random.choice(templates)
        
        # Add a contextual closing if appropriate
        if similarity_score >= 0.7 and random.random() > 0.5:
            closings = [
                " Highly recommend scheduling an interview.",
                " Definitely worth a conversation.",
                " Strong potential for immediate impact.",
                " Could contribute from day one.",
            ]
            summary += random.choice(closings)
        
        return summary
    
    def batch_generate_summaries(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple candidates.
        
        Args:
            candidates: List of candidate dictionaries
            job_description: Job description text
            
        Returns:
            Updated candidates with summaries
        """
        logger.info(f"Generating summaries for {len(candidates)} candidates")
        
        for candidate in candidates:
            try:
                # Generate summary
                summary = self.generate_fit_summary(
                    job_description,
                    candidate['text'],
                    candidate['similarity_score'],
                    candidate.get('matching_skills')
                )
                candidate['fit_summary'] = summary
                
            except Exception as e:
                logger.error(f"Error generating summary for {candidate['candidate_name']}: {e}")
                candidate['fit_summary'] = (
                    f"Candidate shows {candidate['percentage_score']:.1f}% match "
                    f"with the job requirements based on resume analysis."
                )
        
        return candidates
    
    # Helper methods for diverse summary generation
    def _get_strength_statement(self, skills, needs_ml, needs_cloud, has_leadership):
        """Generate strength-based statement."""
        import random
        if needs_ml and any('learning' in s.lower() or 'ai' in s.lower() for s in skills):
            return "Deep ML expertise combined with production deployment experience makes them ideal for your AI initiatives."
        elif needs_cloud and any('aws' in s.lower() or 'cloud' in s.lower() for s in skills):
            return "Proven cloud architecture skills with hands-on experience scaling distributed systems."
        elif has_leadership:
            return "Leadership experience combined with technical depth enables them to drive projects end-to-end."
        else:
            return "Technical proficiency across the full stack with particular strength in implementation and delivery."
    
    def _get_experience_highlight(self, years, has_leadership, has_masters, has_phd):
        """Highlight experience and education."""
        parts = []
        if has_phd:
            parts.append("PhD-level expertise")
        elif has_masters:
            parts.append("Advanced degree")
        if years > 7:
            parts.append(f"{years}+ years of progressive experience")
        elif years > 0:
            parts.append(f"{years} years of hands-on experience")
        if has_leadership:
            parts.append("proven leadership")
        
        if parts:
            return "Brings " + " with ".join(parts) + "."
        return "Brings relevant industry experience."
    
    def _get_skills_statement(self, skills):
        """Create skills-focused statement."""
        import random
        if len(skills) > 5:
            return f"Comprehensive skill set spanning {len(skills)} key technologies your team uses."
        elif len(skills) > 2:
            return f"Proficiency in critical areas including {', '.join(skills[:3])}."
        elif skills:
            return f"Relevant expertise in {' and '.join(skills)}."
        return "Transferable skills that align with role requirements."
    
    def _get_experience_descriptor(self, years, has_leadership):
        """Describe experience level."""
        if years > 10:
            return "decade+ of industry expertise"
        elif years > 5:
            return f"{years} years of progressive technical growth"
        elif has_leadership:
            return "leadership experience and technical acumen"
        else:
            return "solid technical foundation"
    
    def _get_unique_value_prop(self, skills, years, has_leadership, needs_ml, needs_cloud):
        """Generate unique value proposition."""
        import random
        props = []
        if needs_ml and any('learning' in s.lower() for s in skills):
            props.append("Combines ML expertise with production engineering skills")
        if needs_cloud and any('cloud' in s.lower() or 'aws' in s.lower() for s in skills):
            props.append("Cloud-native development experience at scale")
        if years > 5:
            props.append(f"Battle-tested through {years}+ years of real-world challenges")
        if has_leadership:
            props.append("Natural leader who can mentor and grow the team")
        
        if props:
            return random.choice(props) + "."
        return "Brings a unique combination of technical skills and practical experience."
    
    def _get_growth_statement(self, years, has_leadership):
        """Statement about growth potential."""
        if has_leadership:
            return "Track record of taking ownership and delivering results."
        elif years > 3:
            return "Consistent career progression demonstrates adaptability and learning agility."
        else:
            return "Shows strong potential for growth within the role."
    
    def _get_skill_area(self, skills, needs_ml, needs_backend):
        """Identify primary skill area."""
        if needs_ml and skills:
            return "machine learning and data engineering"
        elif needs_backend and skills:
            return "backend development and system design"
        elif len(skills) > 2:
            return f"{skills[0]} and {skills[1]}"
        elif skills:
            return skills[0]
        return "software development"
    
    def _get_potential_statement(self, has_masters, years):
        """Statement about potential."""
        if has_masters:
            return "Advanced education provides strong theoretical foundation for complex problem-solving."
        elif years > 5:
            return "Seasoned professional ready to take on new challenges."
        else:
            return "Demonstrates commitment to continuous learning and improvement."
    
    def _get_fit_assessment(self, score, years, has_leadership):
        """Assess fit level."""
        if score > 0.7:
            return f"With {years if years else 'relevant'} years experience, they're ready to contribute immediately."
        elif has_leadership:
            return "Leadership experience suggests ability to work independently and drive initiatives."
        else:
            return "Shows promise for success in this role with minimal ramp-up time."
    
    def _get_transferable_skills(self, skills, needs_ml, needs_cloud):
        """Identify transferable skills."""
        if skills and len(skills) > 2:
            return f"Skills in {', '.join(skills[:2])} directly transfer to your tech stack."
        elif needs_ml:
            return "Analytical mindset and problem-solving skills align with ML requirements."
        elif needs_cloud:
            return "Understanding of distributed systems translates well to cloud architecture."
        else:
            return "Core competencies provide solid foundation for role-specific growth."
    
    def _get_development_potential(self, years, has_masters):
        """Assess development potential."""
        if has_masters:
            return "Advanced degree indicates strong learning capacity and theoretical knowledge."
        elif years > 2:
            return f"With {years} years of experience, has demonstrated ability to grow and adapt."
        else:
            return "Shows enthusiasm and readiness to develop deeper expertise."
    
    def _get_relevant_areas(self, skills, resume_text):
        """Find relevant areas from resume."""
        if skills and len(skills) >= 2:
            return f"{skills[0]} and {skills[1]}"
        elif 'python' in resume_text.lower():
            return "Python development"
        elif 'java' in resume_text.lower():
            return "Java development"
        else:
            return "software development"
    
    def _get_adjacent_experience(self, skills, years):
        """Describe adjacent experience."""
        if years > 3:
            return f"While coming from a slightly different background, {years} years of technical experience provides valuable perspective."
        elif skills:
            return f"Experience with {skills[0] if skills else 'related technologies'} demonstrates technical aptitude."
        else:
            return "Technical background in related areas shows adaptability."
    
    def _get_growth_trajectory(self, has_leadership, years, skills):
        """Describe growth trajectory."""
        if has_leadership:
            return "Leadership experience indicates high potential for growth into senior roles."
        elif years > 2 and skills:
            return f"Steady skill development in {len(skills)} technologies shows commitment to learning."
        else:
            return "Early career professional with room to grow into the role."
    
    def _get_potential_indicator(self, skills, has_masters):
        """Indicate potential despite lower match."""
        if has_masters:
            return "strong academic foundation and learning ability"
        elif skills and len(skills) > 1:
            return f"aptitude in {skills[0]} that could extend to your requirements"
        else:
            return "potential for development with proper mentorship"
    
    def _get_transferable_value(self, skills, years):
        """Identify transferable value."""
        if years > 5:
            return f"Brings {years} years of professional experience with transferable problem-solving skills."
        elif skills:
            return f"Knowledge of {skills[0] if skills else 'technology'} provides starting point for role-specific training."
        else:
            return "Fresh perspective could benefit team diversity."
    
    def _get_learning_potential(self, has_masters, has_phd, years):
        """Assess learning potential."""
        if has_phd:
            return "PhD demonstrates exceptional research and learning capabilities."
        elif has_masters:
            return "Graduate education shows ability to master complex concepts."
        elif years > 0:
            return f"Has shown ability to acquire skills over {years} years in industry."
        else:
            return "Entry-level enthusiasm with strong potential for development."
    
    def _get_unique_angle(self, skills, resume_text):
        """Find unique angle for low-match candidates."""
        if 'startup' in resume_text.lower():
            return "Startup experience brings agility and versatility."
        elif 'enterprise' in resume_text.lower():
            return "Enterprise experience provides understanding of scale and process."
        elif skills:
            return f"Background in {skills[0] if skills else 'technology'} offers fresh perspective."
        else:
            return "Different background could bring innovative approaches to problems."
