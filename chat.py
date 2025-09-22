# Add these imports to your existing app.py
from typing import List, Union, Any, Optional, Dict, Literal
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for Word documents
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import re
import spacy
from datetime import datetime
import json

# Add these new models to your existing models section

class ChatMessage(BaseModel):
    role: Literal["user", "bot"]
    content: str
    timestamp: datetime

class ChatRequest(BaseModel):
    message: str
    chat_phase: Literal["intro", "profile_building", "job_searching", "results"] = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []

class UserProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience: Optional[int] = None
    education: Optional[str] = None
    location: Optional[str] = None
    preferences: Optional[Dict] = {}
    raw_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    message_type: Literal["text", "cv_upload", "job_results", "profile_summary"] = "text"
    chat_phase: Optional[str] = None
    profile_data: Optional[UserProfile] = None
    jobs: Optional[List[Dict]] = None
    suggestions: Optional[List[str]] = []

class CVUploadResponse(BaseModel):
    response: str
    profile_data: Optional[UserProfile] = None
    success: bool = True

# Add these new service classes

class CVParsingService:
    """Service for parsing CV files and extracting structured information"""
    
    def __init__(self):
        # Load spaCy model for NER (you'll need to install: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def extract_text_from_file(self, file: UploadFile) -> str:
        """Extract text from PDF, DOC, or DOCX files"""
        try:
            content = await file.read()
            
            if file.filename.lower().endswith('.pdf'):
                return self._extract_text_from_pdf(content)
            elif file.filename.lower().endswith(('.doc', '.docx')):
                return self._extract_text_from_docx(content)
            else:
                raise ValueError("Unsupported file format")
                
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX using python-docx"""
        try:
            from io import BytesIO
            doc = docx.Document(BytesIO(content))
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Error processing DOCX: {str(e)}")
    
    async def parse_cv_with_gpt(self, cv_text: str) -> UserProfile:
        """Parse CV text using Azure GPT to extract structured information"""
        
        # Clean and truncate text for GPT processing
        cleaned_text = re.sub(r'\s+', ' ', cv_text).strip()
        if len(cleaned_text) > 4000:
            cleaned_text = cleaned_text[:4000] + "..."
        
        prompt = f"""
Extract structured information from this CV/Resume text. Return ONLY a valid JSON object with these fields:

{{
    "name": "Full name of the candidate",
    "email": "Email address if found",
    "phone": "Phone number if found", 
    "skills": ["List", "of", "technical", "skills"],
    "experience": number_of_years_experience,
    "education": "Highest qualification",
    "location": "City, State/Country if found"
}}

CV Text:
{cleaned_text}

Rules:
1. Extract ALL technical skills mentioned (programming languages, frameworks, tools, etc.)
2. Calculate total years of experience from work history
3. For location, extract current city/state or preferred location
4. Return valid JSON only, no explanation text
5. If information not found, use null or empty array []
"""

        try:
            logger.info("Parsing CV with Azure GPT...")
            
            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert CV parser. Extract structured information and return only valid JSON. No explanatory text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()
            
            try:
                profile_data = json.loads(content)
                
                # Validate and create UserProfile
                user_profile = UserProfile(
                    name=profile_data.get('name'),
                    email=profile_data.get('email'),
                    phone=profile_data.get('phone'),
                    skills=profile_data.get('skills', [])[:20],  # Limit skills
                    experience=profile_data.get('experience'),
                    education=profile_data.get('education'),
                    location=profile_data.get('location'),
                    raw_text=cv_text[:2000]  # Store truncated raw text
                )
                
                logger.info(f"Successfully parsed CV: {len(user_profile.skills)} skills extracted")
                return user_profile
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT CV response: {e}")
                logger.debug(f"Raw response: {content}")
                # Fallback to rule-based extraction
                return self._fallback_cv_parsing(cv_text)
                
        except Exception as e:
            logger.error(f"GPT CV parsing failed: {e}")
            return self._fallback_cv_parsing(cv_text)
    
    def _fallback_cv_parsing(self, cv_text: str) -> UserProfile:
        """Fallback rule-based CV parsing"""
        logger.info("Using fallback rule-based CV parsing")
        
        # Extract basic information using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        emails = re.findall(email_pattern, cv_text)
        phones = re.findall(phone_pattern, cv_text)
        
        # Extract skills using common patterns
        skill_patterns = [
            r'\b(Python|Java|JavaScript|React|Node\.js|Django|Flask|SQL|MongoDB|AWS|Docker|Kubernetes|Git|HTML|CSS|Angular|Vue|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(Machine Learning|AI|Data Science|Deep Learning|TensorFlow|PyTorch|Pandas|NumPy|Scikit-learn)\b',
            r'\b(Azure|GCP|Jenkins|CI/CD|DevOps|Microservices|REST API|GraphQL)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, cv_text, re.IGNORECASE)
            skills.update(matches)
        
        # Extract experience (simple heuristic)
        experience_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
            r'experience[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        experience = None
        for pattern in experience_patterns:
            match = re.search(pattern, cv_text, re.IGNORECASE)
            if match:
                experience = int(match.group(1))
                break
        
        return UserProfile(
            email=emails[0] if emails else None,
            phone=phones[0] if phones else None,
            skills=list(skills)[:15],  # Limit to 15 skills
            experience=experience,
            raw_text=cv_text[:2000]
        )


class ConversationalJobService:
    """Service for handling conversational job matching"""
    
    @staticmethod
    async def handle_chat_message(request: ChatRequest) -> ChatResponse:
        """Handle conversational messages and manage chat flow"""
        
        message = request.message.lower().strip()
        chat_phase = request.chat_phase
        user_profile = request.user_profile or {}
        
        try:
            # Determine intent and generate response
            if chat_phase == "intro":
                return await ConversationalJobService._handle_intro_phase(message, user_profile)
            elif chat_phase == "profile_building":
                return await ConversationalJobService._handle_profile_building(message, user_profile, request.conversation_history)
            elif chat_phase == "job_searching":
                return await ConversationalJobService._handle_job_search(message, user_profile)
            else:
                return ChatResponse(
                    response="I'm here to help you find jobs! What would you like to know?",
                    message_type="text"
                )
                
        except Exception as e:
            logger.error(f"Error in conversational service: {e}")
            return ChatResponse(
                response="I encountered an error. Let me help you in a different way. What information can you share about your background?",
                message_type="text",
                chat_phase="profile_building"
            )
    
    @staticmethod
    async def _handle_intro_phase(message: str, user_profile: Dict) -> ChatResponse:
        """Handle introduction phase - CV upload or chat choice"""
        
        if any(keyword in message for keyword in ["upload", "cv", "resume", "file"]):