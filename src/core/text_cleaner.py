"""
Text cleaning and preprocessing utilities.
"""

import re
from typing import Optional, List, Dict
from loguru import logger


class TextCleaner:
    """Clean and preprocess text for embedding generation."""
    
    def __init__(self):
        """Initialize the text cleaner."""
        self.min_length = 50
        self.max_length = 10000
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
            
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaned = cleaner.clean_text("  Some\\n\\ntext  with   spaces  ")
            >>> print(cleaned)
        """
        if not text:
            return ""
            
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
            
            # Remove multiple consecutive punctuation marks
            text = re.sub(r'([\.,:;!?])\1+', r'\1', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            # Truncate if too long
            if len(text) > self.max_length:
                logger.warning(f"Text truncated from {len(text)} to {self.max_length} characters")
                text = text[:self.max_length]
                
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def extract_candidate_name(self, text: str, filename: Optional[str] = None) -> str:
        """
        Extract candidate name from resume text or filename.
        
        Args:
            text: Resume text
            filename: Optional filename
            
        Returns:
            Extracted name or generated ID
        """
        # Try to extract from filename first
        if filename:
            # Remove extension and clean filename
            name = filename.rsplit('.', 1)[0]
            name = re.sub(r'[_\-]', ' ', name)
            name = re.sub(r'resume|cv|curriculum|vitae', '', name, flags=re.IGNORECASE)
            name = name.strip()
            if name and len(name) > 2:
                return name.title()
        
        # Try to extract from text (look for name patterns at the beginning)
        lines = text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            # Simple heuristic: if line is 2-4 words and contains mostly letters
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word.replace('-', '').isalpha() for word in words):
                    return ' '.join(words).title()
        
        # Fallback to generated ID
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"Candidate_{text_hash}"
    
    def extract_key_skills(self, text: str) -> List[str]:
        """
        Extract key skills from text.
        
        Args:
            text: Resume or job description text
            
        Returns:
            List of extracted skills
        """
        # Common skill keywords (expandable)
        skill_patterns = [
            r'python', r'java\b', r'javascript', r'typescript', r'react', r'angular',
            r'node\.?js', r'django', r'flask', r'fastapi', r'sql', r'nosql', r'mongodb',
            r'postgresql', r'mysql', r'docker', r'kubernetes', r'aws', r'azure', r'gcp',
            r'machine learning', r'deep learning', r'tensorflow', r'pytorch', r'scikit-learn',
            r'pandas', r'numpy', r'data science', r'nlp', r'computer vision', r'ai\b',
            r'rest api', r'graphql', r'microservices', r'ci/cd', r'git', r'agile', r'scrum',
            r'leadership', r'communication', r'problem solving', r'teamwork'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for pattern in skill_patterns:
            if re.search(pattern, text_lower):
                # Clean up the skill name
                skill = pattern.replace(r'\b', '').replace(r'\.?', '.').replace('\\', '')
                found_skills.append(skill.title())
        
        return list(set(found_skills))[:10]  # Return top 10 unique skills
    
    def prepare_for_embedding(self, text: str) -> str:
        """
        Prepare text specifically for embedding generation.
        
        Args:
            text: Text to prepare
            
        Returns:
            Prepared text for embedding
        """
        # Clean the text
        text = self.clean_text(text)
        
        # Remove very short text
        if len(text) < self.min_length:
            logger.warning(f"Text too short ({len(text)} chars), may affect quality")
        
        return text
    
    def extract_contact_details(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract detailed contact information from text.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with contact details
        """
        import re
        
        contact = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'location': None,
            'website': None
        }
        
        # Email extraction (improved)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            # Filter out common non-personal emails
            personal_emails = [e for e in emails if not any(x in e.lower() for x in ['noreply', 'support', 'info@', 'admin@'])]
            contact['email'] = personal_emails[0] if personal_emails else emails[0]
        
        # Phone extraction (comprehensive)
        phone_patterns = [
            r'(?:(?:Tel|Phone|Mobile|Cell)[\s:]*)?(?:[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?)?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
            r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text, re.IGNORECASE)
            if phones:
                # Clean and validate
                for phone in phones:
                    cleaned = re.sub(r'[^\d+]', '', phone)
                    if 10 <= len(cleaned) <= 15:  # Valid phone length
                        contact['phone'] = phone.strip()
                        break
                if contact['phone']:
                    break
        
        # LinkedIn extraction
        linkedin_patterns = [
            r'linkedin\.com/in/([a-zA-Z0-9\-]+)',
            r'LinkedIn[\s:]+([a-zA-Z0-9\-]+)',
            r'linkedin\.com/pub/([a-zA-Z0-9\-]+)',
        ]
        
        for pattern in linkedin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                linkedin_id = match.group(1)
                contact['linkedin'] = f"linkedin.com/in/{linkedin_id}"
                break
        
        # GitHub extraction
        github_pattern = r'github\.com/([a-zA-Z0-9\-]+)'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact['github'] = f"github.com/{github_match.group(1)}"
        
        # Location extraction
        # Common patterns: "City, State" or "City, Country"
        location_indicators = ['Location', 'Address', 'Based in', 'Lives in', 'Residing']
        for indicator in location_indicators:
            pattern = f'{indicator}[\s:]*([A-Za-z\\s]+(?:,\\s*[A-Za-z\\s]+)?)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up location
                if len(location) > 3 and ',' in location:
                    contact['location'] = location
                    break
        
        # If no location found with indicators, try common city patterns
        if not contact['location']:
            # Look for "City, ST" pattern (US cities)
            us_city_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b'
            match = re.search(us_city_pattern, text)
            if match:
                contact['location'] = match.group(1)
        
        # Website extraction
        website_pattern = r'(?:website|portfolio|www)[\s:]*(?:https?://)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)'
        website_match = re.search(website_pattern, text, re.IGNORECASE)
        if website_match:
            contact['website'] = website_match.group(1)
        
        return contact
