import os
import json
import re
import asyncio
import pickle
from typing import List, Union, Any, Optional, Dict
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import asyncpg
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss


from fastapi import File, UploadFile
from typing import Literal
import fitz  # PyMuPDF - you'll need: pip install PyMuPDF
import docx

from cv import CVProcessor, CVProfile
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive embedding operations
embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

# Azure OpenAI client for text generation only
try:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint and not endpoint.endswith('/'):
        endpoint = endpoint + '/'
    
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=endpoint
    )
    
    gpt_deployment = os.getenv("AZURE_GPT_DEPLOYMENT")
    if not gpt_deployment:
        raise ValueError("Missing AZURE_GPT_DEPLOYMENT in environment variables")
        
    logger.info(f"Initialized Azure client for GPT: {gpt_deployment}")
    
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise

# PostgreSQL connection string
DB_URL = os.getenv("DATABASE_URL")


class ChatRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    response: str
    message_type: str = "text"
    chat_phase: Optional[str] = None
    profile_data: Optional[Dict] = None
    jobs: Optional[List[Dict]] = None
    suggestions: Optional[List[str]] = []

class CVUploadResponse(BaseModel):
    response: str
    profile_data: Optional[Dict] = None
    success: bool = True

class CVUploadRequest(BaseModel):
    """CV upload request model"""
    pass

class ChatWithCVRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []
    cv_profile_data: Optional[Dict] = None

class CVAnalysisResponse(BaseModel):
    """Enhanced CV analysis response"""
    success: bool
    message: str
    profile: Optional[Dict[str, Any]] = None
    jobs: Optional[List[Dict]] = None
    total_jobs_found: int = 0
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    recommendations: Optional[List[str]] = None


class SimpleChatService:
    """Simple chat service for basic conversation and manual skill entry"""
    
    @staticmethod
    async def handle_chat_message(request: ChatRequest) -> ChatResponse:
        """Handle chat messages with basic logic"""
        
        message = request.message.lower().strip()
        chat_phase = request.chat_phase
        user_profile = request.user_profile or {}
        
        try:
            if chat_phase == "intro":
                if any(word in message for word in ["upload", "cv", "resume", "file"]):
                    return ChatResponse(
                        response="Great! Please click the paperclip icon to upload your CV. I support PDF, DOC, and DOCX files.",
                        message_type="text",
                        chat_phase="intro"
                    )
                elif any(word in message for word in ["chat", "talk", "build", "skills"]):
                    return ChatResponse(
                        response="Perfect! Let's build your profile together. What are your main technical skills? (e.g., Python, React, SQL)",
                        message_type="text",
                        chat_phase="profile_building"
                    )
                else:
                    return ChatResponse(
                        response="I can help you find jobs in two ways:\n\n1. 📄 Upload your CV - I'll analyze it automatically\n2. 💬 Chat with me - I'll ask about your skills\n\nWhich would you prefer?",
                        message_type="text",
                        chat_phase="intro",
                        suggestions=["Upload CV", "Let's chat", "Tell me about skills"]
                    )
            
            elif chat_phase == "profile_building":
                # Extract skills from message
                skills = []
                skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'html', 'css', 'node', 'angular', 'vue', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'aws', 'docker', 'kubernetes']
                
                for skill in skill_keywords:
                    if skill in message:
                        skills.append(skill.title())
                
                if skills:
                    # Use your existing job search
                    try:
                        skills_text = " ".join(skills)
                        skills_embedding = await embedding_service.get_embedding(skills_text)
                        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=20)
                        
                        if similar_jobs:
                            ranked_jobs = await gpt_service.rerank_jobs(skills, similar_jobs)
                            job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                            complete_jobs = await get_complete_job_details(job_ids)
                            
                            # Format jobs for chat response
                            job_results = []
                            for job_data in ranked_jobs[:5]:
                                complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                                if complete_job:
                                    job_results.append({
                                        "ncspjobid": job_data["ncspjobid"],
                                        "title": job_data["title"],
                                        "organization_name": complete_job.get("organization_name", ""),
                                        "match_percentage": job_data["match_percentage"],
                                        "statename": complete_job.get("statename", ""),
                                        "districtname": complete_job.get("districtname", ""),
                                        "avewage": complete_job.get("avewage", 0),
                                        "aveexp": complete_job.get("aveexp", 0)
                                    })
                            
                            return ChatResponse(
                                response=f"Great! I found {len(job_results)} jobs matching your skills: {', '.join(skills)}. Here are the top matches:",
                                message_type="job_results",
                                chat_phase="job_searching",
                                profile_data={"skills": skills},
                                jobs=job_results
                            )
                        else:
                            return ChatResponse(
                                response="I understand your skills, but let me search more broadly. What's your experience level in years?",
                                message_type="text",
                                chat_phase="profile_building"
                            )
                    except Exception as e:
                        logger.error(f"Job search failed in chat: {e}")
                        return ChatResponse(
                            response="I noted your skills. What other technical skills do you have?",
                            message_type="text",
                            chat_phase="profile_building"
                        )
                else:
                    return ChatResponse(
                        response="I'd like to help you find jobs. Please tell me your technical skills. For example: 'I know Python, React, and SQL'",
                        message_type="text",
                        chat_phase="profile_building"
                    )
            
            else:
                return ChatResponse(
                    response="I can help you find more jobs or refine your search. What would you like to do?",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Show more jobs", "Different skills", "Start over"]
                )
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ChatResponse(
                response="Let me help you find jobs. What technical skills do you have?",
                message_type="text",
                chat_phase="profile_building"
            )

# Add enhanced service specifically for CV-based interactions
class CVChatService:
    """Enhanced chat service specifically for CV upload interactions"""
    
    @staticmethod
    async def handle_cv_upload_chat(cv_profile: CVProfile) -> ChatResponse:
        """Handle chat after CV upload with enhanced profile data"""
        try:
            if cv_profile.skills and len(cv_profile.skills) >= 3:
                # Search for jobs using extracted skills
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=15)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    # Format jobs for chat
                    job_results = []
                    for job_data in ranked_jobs[:5]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0)
                            })
                    
                    profile_summary = {
                        "name": cv_profile.name,
                        "skills": cv_profile.skills[:10],
                        "experience_count": len(cv_profile.experience),
                        "confidence": cv_profile.confidence_score
                    }
                    
                    return ChatResponse(
                        response=f"🎉 Perfect! I've analyzed your CV and found your skills: {', '.join(cv_profile.skills[:5])}{'...' if len(cv_profile.skills) > 5 else ''}. Here are {len(job_results)} matching jobs:",
                        message_type="cv_results",
                        chat_phase="job_results",
                        profile_data=profile_summary,
                        jobs=job_results
                    )
                else:
                    return ChatResponse(
                        response=f"I've extracted your skills: {', '.join(cv_profile.skills[:5])}. Let me search for more opportunities or we can refine your profile.",
                        message_type="text",
                        chat_phase="profile_refinement",
                        profile_data={"skills": cv_profile.skills}
                    )
            else:
                return ChatResponse(
                    response="I've processed your CV but found limited technical skills. Let's chat to build a complete profile for better job matching.",
                    message_type="text", 
                    chat_phase="profile_building"
                )
                
        except Exception as e:
            logger.error(f"CV chat integration failed: {e}")
            return ChatResponse(
                response="I've processed your CV! Let's discuss your skills to find the best job matches.",
                message_type="text",
                chat_phase="profile_building"
            )
    
    @staticmethod
    async def handle_cv_followup_chat(request: ChatRequest, cv_profile: CVProfile) -> ChatResponse:
        """Handle follow-up chat after CV analysis"""
        message = request.message.lower().strip()
        
        try:
            # Handle requests for more jobs
            if any(word in message for word in ["more jobs", "show more", "additional", "other jobs"]):
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=25)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    # Skip first 5 jobs (already shown) and get next 5
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[5:10]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    job_results = []
                    for job_data in ranked_jobs[5:10]:
                        complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                        if complete_job:
                            job_results.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0)
                            })
                    
                    return ChatResponse(
                        response=f"Here are {len(job_results)} more job opportunities based on your CV:",
                        message_type="job_results",
                        chat_phase="job_results",
                        jobs=job_results
                    )
            
            # Handle skill refinement requests
            elif any(word in message for word in ["add skill", "more skill", "also know", "i can"]):
                # Extract additional skills from message
                additional_skills = []
                skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'html', 'css', 'node', 'angular', 'vue', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'aws', 'docker', 'kubernetes']
                
                for skill in skill_keywords:
                    if skill in message and skill.title() not in cv_profile.skills:
                        additional_skills.append(skill.title())
                
                if additional_skills:
                    combined_skills = cv_profile.skills + additional_skills
                    skills_text = " ".join(combined_skills)
                    skills_embedding = await embedding_service.get_embedding(skills_text)
                    similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=20)
                    
                    if similar_jobs:
                        ranked_jobs = await gpt_service.rerank_jobs(combined_skills, similar_jobs)
                        job_ids = [job["ncspjobid"] for job in ranked_jobs[:5]]
                        complete_jobs = await get_complete_job_details(job_ids)
                        
                        job_results = []
                        for job_data in ranked_jobs[:5]:
                            complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
                            if complete_job:
                                job_results.append({
                                    "ncspjobid": job_data["ncspjobid"],
                                    "title": job_data["title"],
                                    "organization_name": complete_job.get("organization_name", ""),
                                    "match_percentage": job_data["match_percentage"],
                                    "statename": complete_job.get("statename", ""),
                                    "districtname": complete_job.get("districtname", ""),
                                    "avewage": complete_job.get("avewage", 0),
                                    "aveexp": complete_job.get("aveexp", 0)
                                })
                        
                        return ChatResponse(
                            response=f"Great! I've added {', '.join(additional_skills)} to your profile. Here are updated job matches:",
                            message_type="job_results",
                            chat_phase="job_results",
                            profile_data={"skills": combined_skills},
                            jobs=job_results
                        )
                else:
                    return ChatResponse(
                        response="What additional skills would you like to add to your profile? For example: 'I also know Docker and AWS'",
                        message_type="text",
                        chat_phase="profile_refinement"
                    )
            
            # Default response
            else:
                return ChatResponse(
                    response="I can help you with:\n• Show more job opportunities\n• Add skills to your profile\n• Search by location\n• Start a new search\n\nWhat would you like to do?",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Show more jobs", "Add skills", "Search by location", "Start over"]
                )
                
        except Exception as e:
            logger.error(f"CV followup chat failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs or refine your search. What would you like to do?",
                message_type="text",
                chat_phase="job_results"
            )


# Initialize the chat service
chat_service = SimpleChatService()
cv_chat_service = CVChatService()


# Request/Response models
class JobSearchRequest(BaseModel):
    skills: List[str] = Field(..., min_items=1, max_items=50, description="List of skills to search for")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of jobs to return")
    
    @validator('skills', pre=True)
    def validate_and_clean_skills(cls, v):
        if not v:
            raise ValueError('Skills list cannot be empty')
        
        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
        # Filter and clean skills
        cleaned_skills = []
        for skill in v:
            if isinstance(skill, str) and skill.strip():
                # Remove special characters but keep programming symbols
                cleaned = re.sub(r'[^\w\s+#.-]', '', skill.strip())
                if cleaned and len(cleaned) > 1:
                    cleaned_skills.append(cleaned)
        
        if not cleaned_skills:
            raise ValueError('No valid skills found after cleaning')
        
        if len(cleaned_skills) > 50:
            cleaned_skills = cleaned_skills[:50]
        
        return cleaned_skills
    
class CourseRecommendationRequest(BaseModel):
    keywords_unmatched: List[str] = Field(..., min_items=1, max_items=20, description="List of unmatched keywords to get course recommendations for")
    
    @validator('keywords_unmatched', pre=True)
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        
        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
        # Clean keywords
        cleaned_keywords = []
        for keyword in v:
            if isinstance(keyword, str) and keyword.strip():
                cleaned = keyword.strip()
                if len(cleaned) > 1:
                    cleaned_keywords.append(cleaned)
        
        if not cleaned_keywords:
            raise ValueError('No valid keywords found after cleaning')
        
        if len(cleaned_keywords) > 20:
            cleaned_keywords = cleaned_keywords[:20]
        
        return cleaned_keywords
    
class CourseRecommendation(BaseModel):
    course_name: str
    platform: str
    duration: str
    link: str
    educator: str
    skill_covered: str
    difficulty_level: Optional[str] = None
    rating: Optional[str] = None

class CourseRecommendationResponse(BaseModel):
    recommendations: List[CourseRecommendation]
    keywords_processed: List[str]
    total_recommendations: int
    processing_time_ms: int

class JobResult(BaseModel):
    ncspjobid: str
    title: str
    match_percentage: float = Field(..., ge=0, le=100)
    similarity_score: Optional[float] = None
    keywords: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    organizationid: Optional[int] = None
    organization_name: Optional[str] = None
    numberofopenings: Optional[int] = None
    industryname: Optional[str] = None
    sectorname: Optional[str] = None
    functionalareaname: Optional[str] = None
    functionalrolename: Optional[str] = None
    aveexp: Optional[float] = None
    avewage: Optional[float] = None
    gendercode: Optional[str] = None
    highestqualification: Optional[str] = None
    statename: Optional[str] = None
    districtname: Optional[str] = None
    keywords_matched: Optional[List[str]] = None
    keywords_unmatched: Optional[List[str]] = None
    user_skills_matched: Optional[List[str]] = None
    keyword_match_score: Optional[float] = None

class JobSearchResponse(BaseModel):
    jobs: List[JobResult]
    query_skills: List[str]
    total_found: int
    processing_time_ms: int

class LocalEmbeddingService:
    """Local embedding service using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            test_embedding = self.model.encode("test input", convert_to_tensor=False)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation for thread pool execution"""
        try:
            if not self.model:
                raise ValueError("Model not initialized")
            
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise
    
    async def get_embedding(self, text: Union[str, List[str], Any]) -> List[float]:
        """Generate embedding using local Sentence Transformer model"""
        
        # Normalize input to string
        try:
            if isinstance(text, list):
                processed_text = " ".join(str(item) for item in text if item)
            elif isinstance(text, (int, float)):
                processed_text = str(text)
            elif text is None:
                raise ValueError("Embedding input cannot be None")
            else:
                processed_text = str(text)
            
            processed_text = re.sub(r'\s+', ' ', processed_text.strip())
            
            if not processed_text or len(processed_text) == 0:
                raise ValueError("Embedding input must be non-empty after processing")
            
            if len(processed_text) > 2000:
                processed_text = processed_text[:2000]
            
        except Exception as e:
            logger.error(f"Input processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                embedding_executor,
                self._generate_embedding_sync,
                processed_text
            )
            
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding generated")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

class FAISSVectorStore:
    """FAISS-based vector store for job similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.job_metadata = []
        self.is_loaded = False
        self._lock = threading.Lock()
        self.index_file = "faiss_job_index.bin"
        self.metadata_file = "job_metadata.pkl"
    
    async def load_jobs_from_db(self, force_reload: bool = False):
        """Load jobs from PostgreSQL and build/load FAISS index"""
        if self.is_loaded and not force_reload:
            logger.info("FAISS index already loaded")
            return
        
        with self._lock:
            try:
                # Try to load existing index first
                if not force_reload and os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                    logger.info("Loading existing FAISS index from disk...")
                    self.index = faiss.read_index(self.index_file)
                    
                    with open(self.metadata_file, 'rb') as f:
                        self.job_metadata = pickle.load(f)
                    
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} jobs")
                    self.is_loaded = True
                    return
                
                logger.info("Building new FAISS index from database...")
                
                # Connect to database and fetch jobs
                conn = await asyncpg.connect(DB_URL)
                try:
                    rows = await conn.fetch("""
                        SELECT ncspjobid, title, keywords, description
                        FROM vacancies_summary
                        WHERE (keywords IS NOT NULL AND keywords != '') 
                           OR (description IS NOT NULL AND description != '')
                        ORDER BY ncspjobid;
                    """)
                    
                    if not rows:
                        logger.warning("No jobs found in database")
                        return
                    
                    logger.info(f"Found {len(rows)} jobs in database")
                    
                    # Prepare job texts and metadata
                    job_texts = []
                    self.job_metadata = []
                    
                    for row in rows:
                        # Combine title, keywords, and description for embedding
                        text_parts = []
                        if row['title']:
                            text_parts.append(row['title'])
                        if row['keywords']:
                            text_parts.append(row['keywords'])
                        if row['description']:
                            desc = row['description'][:500] if row['description'] else ""
                            if desc:
                                text_parts.append(desc)
                        
                        job_text = " ".join(text_parts)
                        job_texts.append(job_text)
                        
                        self.job_metadata.append({
                            'ncspjobid': row['ncspjobid'],
                            'title': row['title'],
                            'keywords': row['keywords'],
                            'description': row['description']
                        })
                    
                    # Generate embeddings for all jobs
                    logger.info("Generating embeddings for all jobs...")
                    embeddings = await self._generate_job_embeddings(job_texts)
                    
                    # Create FAISS index
                    self.index = faiss.IndexFlatIP(self.dimension)
                    
                    # Add embeddings to index
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)
                    
                    # Save index and metadata to disk
                    faiss.write_index(self.index, self.index_file)
                    with open(self.metadata_file, 'wb') as f:
                        pickle.dump(self.job_metadata, f)
                    
                    logger.info(f"Built FAISS index with {self.index.ntotal} jobs and saved to disk")
                    self.is_loaded = True
                    
                finally:
                    await conn.close()
                    
            except Exception as e:
                logger.error(f"Failed to load jobs into FAISS: {e}")
                raise HTTPException(status_code=503, detail="Failed to initialize job search index")
    
    async def _generate_job_embeddings(self, job_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for job texts using the embedding service"""
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(job_texts), batch_size):
            batch = job_texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(job_texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                try:
                    embedding = await embedding_service.get_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for job text: {e}")
                    batch_embeddings.append([0.0] * self.dimension)
            
            embeddings.extend(batch_embeddings)
            await asyncio.sleep(0.1)
        
        return embeddings
    
    async def search_similar_jobs(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        """Search for similar jobs using FAISS"""
        if not self.is_loaded or self.index is None:
            raise HTTPException(status_code=503, detail="Job search index not available")
        
        try:
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0:
                    job_data = self.job_metadata[idx].copy()
                    job_data['similarity'] = float(similarity)
                    results.append(job_data)
            
            logger.info(f"FAISS search returned {len(results)} similar jobs")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise HTTPException(status_code=500, detail="Job search failed")


class CourseRecommendationService:
    """Service class for course recommendations using Azure OpenAI"""
    
    @staticmethod
    async def get_course_recommendations(keywords: List[str]) -> List[Dict]:
        """Get course recommendations for unmatched keywords using Azure OpenAI GPT"""
        
        if not keywords:
            return []
        
        keywords_str = ', '.join(keywords)
        
        prompt = f"""
    You are an expert career advisor. For each of these technical skills: {keywords_str}

    Provide EXACTLY 5 specific, real courses for each skill from these platforms ONLY:
    - Udemy.com
    - Coursera.org  
    - edX.org
    - Pluralsight.com
    - LinkedIn Learning
    - DataCamp.com

    STRICT REQUIREMENTS:
    1. NO generic course names like "Learn Python" or "Master React"
    2. NO Google search links or google.com URLs
    3. Use ACTUAL course titles from real platforms
    4. Each course must have a realistic platform-specific URL
    5. EXACTLY 5 courses per keyword (total: {len(keywords) * 5} courses)

    For each course provide:
    - course_name: Specific real course title
    - platform: One of the approved platforms above
    - duration: Realistic timeframe
    - link: Actual course URL (not search links)
    - educator: Real instructor/organization name
    - skill_covered: The exact keyword from the list
    - difficulty_level: Beginner/Intermediate/Advanced
    - rating: Realistic rating like "4.5/5"

    Example format:
    {{
        "course_name": "Python for Everybody Specialization",
        "platform": "Coursera", 
        "duration": "8 months",
        "link": "https://www.coursera.org/specializations/python",
        "educator": "University of Michigan",
        "skill_covered": "Python",
        "difficulty_level": "Beginner",
        "rating": "4.8/5"
    }}

    Return ONLY a valid JSON array with {len(keywords) * 5} courses total.
    NO explanatory text. NO markdown formatting.
    """

        try:
            logger.info(f"Getting course recommendations for keywords: {keywords_str}")
            
            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a learning advisor. Return ONLY valid JSON array with exactly {len(keywords) * 5} course recommendations. NO explanation text. NO markdown. NO generic course names. NO Google links."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=4000   # Increased for more courses
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response more thoroughly
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = re.sub(r'^[^[]*', '', content)  # Remove text before JSON array
            content = re.sub(r'[^}]*$', '}]', content)  # Ensure proper JSON ending
            content = content.strip()
            
            try:
                return CourseRecommendationService._get_fallback_recommendations(keywords)
                # course_recommendations = json.loads(content)
                # if isinstance(course_recommendations, list):
                #     # Validate each course has required fields
                #     valid_courses = []
                #     for course in course_recommendations:
                #         if (isinstance(course, dict) and 
                #             'course_name' in course and 
                #             'platform' in course and
                #             'link' in course and
                #             'google.com' not in course.get('link', '').lower() and
                #             len(course.get('course_name', '')) > 10):  # Avoid generic names
                #             valid_courses.append(course)
                    
                #     if len(valid_courses) >= len(keywords) * 3:  # At least 3 per keyword
                #         logger.info(f"Successfully got {len(valid_courses)} valid course recommendations")
                #         return valid_courses[:len(keywords) * 5]  # Cap at 5 per keyword
                #     else:
                #         logger.warning(f"Only got {len(valid_courses)} valid courses, expected {len(keywords) * 5}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT course response: {e}")
                logger.debug(f"Raw response: {content}")
            
            # Fallback to sample recommendations
            logger.info("Using fallback sample course recommendations")
            return CourseRecommendationService._get_fallback_recommendations(keywords)
            
        except Exception as e:
            logger.error(f"Course recommendation failed: {e}")
            return CourseRecommendationService._get_fallback_recommendations(keywords)
    
    @staticmethod
    def _get_fallback_recommendations(keywords: List[str]) -> List[Dict]:
        """Provide curated course recommendations with exactly 5 courses per keyword"""
        
        # Comprehensive course database with 5 courses per skill
        course_database = {
            "Python": [
                {
                    "course_name": "Python for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "8 months", 
                    "link": "https://www.coursera.org/specializations/python",
                    "educator": "University of Michigan",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "Complete Python Bootcamp From Zero to Hero",
                    "platform": "Udemy",
                    "duration": "22 hours",
                    "link": "https://www.udemy.com/course/complete-python-bootcamp/", 
                    "educator": "Jose Portilla",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Python Programming Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/python-fundamentals",
                    "educator": "Austin Bingham",
                    "skill_covered": "Python", 
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Computer Science and Programming Using Python",
                    "platform": "edX",
                    "duration": "9 weeks",
                    "link": "https://www.edx.org/course/introduction-to-computer-science-and-programming-7",
                    "educator": "MIT",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner", 
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/python-essential-training-2",
                    "educator": "Bill Weinman",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "JavaScript": [
                {
                    "course_name": "JavaScript Algorithms and Data Structures",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/learn/javascript-algorithms-data-structures",
                    "educator": "University of California San Diego",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "The Complete JavaScript Course 2024: From Zero to Expert!",
                    "platform": "Udemy", 
                    "duration": "69 hours",
                    "link": "https://www.udemy.com/course/the-complete-javascript-course/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "JavaScript Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/javascript-fundamentals",
                    "educator": "Liam McLennan",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to JavaScript",
                    "platform": "edX",
                    "duration": "6 weeks", 
                    "link": "https://www.edx.org/course/introduction-to-javascript",
                    "educator": "W3C",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "JavaScript Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "5 hours",
                    "link": "https://www.linkedin.com/learning/javascript-essential-training",
                    "educator": "Morten Rand-Hendriksen",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "React": [
                {
                    "course_name": "React - The Complete Guide (incl Hooks, React Router, Redux)",
                    "platform": "Udemy", 
                    "duration": "40.5 hours",
                    "link": "https://www.udemy.com/course/react-the-complete-guide-incl-hooks-react-router-redux/",
                    "educator": "Maximilian Schwarzmüller",
                    "skill_covered": "React",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Frontend Development using React Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/react", 
                    "educator": "The Hong Kong University of Science and Technology",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "React.js Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "2 hours", 
                    "link": "https://www.linkedin.com/learning/react-js-essential-training",
                    "educator": "Eve Porcello",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "React Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/react-fundamentals-update",
                    "educator": "Liam McLennan", 
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to ReactJS",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-reactjs",
                    "educator": "Microsoft",
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                }
            ],
            "HTML/CSS": [
                {
                    "course_name": "HTML, CSS, and Javascript for Web Developers",
                    "platform": "Coursera",
                    "duration": "5 weeks",
                    "link": "https://www.coursera.org/learn/html-css-javascript-for-web-developers",
                    "educator": "Johns Hopkins University",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Build Responsive Real-World Websites with HTML and CSS",
                    "platform": "Udemy",
                    "duration": "37.5 hours",
                    "link": "https://www.udemy.com/course/design-and-develop-a-killer-website-with-html5-and-css3/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner to Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "HTML5 and CSS3 Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/html-css-fundamentals",
                    "educator": "Matt Milner",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to Web Development",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-web-development",
                    "educator": "University of California Davis",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "CSS Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/css-essential-training-3",
                    "educator": "Christina Truong",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "SQL": [
                {
                    "course_name": "Introduction to Structured Query Language (SQL)",
                    "platform": "Coursera",
                    "duration": "4 weeks",
                    "link": "https://www.coursera.org/learn/intro-sql",
                    "educator": "University of Michigan",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "The Complete SQL Bootcamp: Go from Zero to Hero",
                    "platform": "Udemy",
                    "duration": "9 hours",
                    "link": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "SQL Server Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/sql-server-fundamentals",
                    "educator": "Pinal Dave",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Databases: Introduction to Databases and SQL Querying",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/databases-introduction-databases-sql",
                    "educator": "IBM",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "SQL Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/sql-essential-training-3",
                    "educator": "Walter Shields",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Django": [
                {
                    "course_name": "Django for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/django",
                    "educator": "University of Michigan", 
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Django - The Practical Guide",
                    "platform": "Udemy",
                    "duration": "23 hours",
                    "link": "https://www.udemy.com/course/python-django-the-practical-guide/",
                    "educator": "Maximilian Schwarzmüller",
                    "skill_covered": "Django", 
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Django Fundamentals",
                    "platform": "Pluralsight", 
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/django-fundamentals-update",
                    "educator": "Reindert-Jan Ekker",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Django Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/django-essential-training",
                    "educator": "Nick Walter",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate", 
                    "rating": "4.6/5"
                },
                {
                    "course_name": "CS50's Web Programming with Python and JavaScript",
                    "platform": "edX",
                    "duration": "12 weeks",
                    "link": "https://www.edx.org/course/cs50s-web-programming-with-python-and-javascript",
                    "educator": "Harvard University",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.8/5"
                }
            ],
            "Spring Boot": [
                {
                    "course_name": "Spring Boot Microservices and Spring Cloud",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/spring-boot-cloud",
                    "educator": "LearnQuest",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Spring Boot For Beginners",
                    "platform": "Udemy",
                    "duration": "7 hours",
                    "link": "https://www.udemy.com/course/spring-boot-tutorial-for-beginners/",
                    "educator": "in28Minutes Official",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Spring Boot Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours",
                    "link": "https://www.pluralsight.com/courses/spring-boot-fundamentals",
                    "educator": "Dan Bunker",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Spring Boot Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/spring-boot-essential-training",
                    "educator": "Frank Moley",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Spring Boot",
                    "platform": "edX",
                    "duration": "4 weeks",
                    "link": "https://www.edx.org/course/introduction-to-spring-boot",
                    "educator": "Microsoft",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.3/5"
                }
            ],
            "Data Analysis": [
                {
                    "course_name": "Google Data Analytics Professional Certificate",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/professional-certificates/google-data-analytics",
                    "educator": "Google",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python for Data Science and Machine Learning Bootcamp",
                    "platform": "Udemy",
                    "duration": "25 hours",
                    "link": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Data Analysis Fundamentals with Tableau",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/tableau-data-analysis-fundamentals",
                    "educator": "Ben Sullins",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Data Analysis using Excel",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/introduction-to-data-analysis-using-excel",
                    "educator": "Rice University",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Excel Essential Training (Microsoft 365)",
                    "platform": "LinkedIn Learning",
                    "duration": "6 hours",
                    "link": "https://www.linkedin.com/learning/excel-essential-training-microsoft-365",
                    "educator": "Dennis Taylor",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Power BI": [
                {
                    "course_name": "Microsoft Power BI Data Analyst Professional Certificate",
                    "platform": "Coursera", 
                    "duration": "5 months",
                    "link": "https://www.coursera.org/professional-certificates/microsoft-power-bi-data-analyst",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Microsoft Power BI Desktop for Business Intelligence",
                    "platform": "Udemy",
                    "duration": "20 hours",
                    "link": "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/",
                    "educator": "Maven Analytics",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Power BI Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours", 
                    "link": "https://www.pluralsight.com/courses/power-bi-fundamentals",
                    "educator": "Stacia Varga",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Analyzing and Visualizing Data with Power BI",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/analyzing-and-visualizing-data-with-power-bi",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Power BI Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/power-bi-essential-training-3",
                    "educator": "Gini Courter",
                    "skill_covered": "Power BI", 
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                }
            ]
        }
        
        recommendations = []
        for keyword in keywords:
            keyword_normalized = keyword.strip().title()
            if keyword_normalized in course_database:
                recommendations.extend(course_database[keyword_normalized])
            else:
                # Fallback for unknown skills - still no generic names
                skill_lower = keyword.lower().replace(' ', '-')
                recommendations.extend([
                    {
                        "course_name": f"Complete {keyword} Development Masterclass",
                        "platform": "Udemy",
                        "duration": "15 hours",
                        "link": f"https://www.udemy.com/topic/{skill_lower}/",
                        "educator": "Expert Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "All Levels", 
                        "rating": "4.5/5"
                    },
                    {
                        "course_name": f"{keyword} Fundamentals",
                        "platform": "Pluralsight",
                        "duration": "6 hours",
                        "link": f"https://www.pluralsight.com/courses/{skill_lower}-fundamentals",
                        "educator": "Industry Expert",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.4/5"
                    },
                    {
                        "course_name": f"{keyword} Essential Training",
                        "platform": "LinkedIn Learning",
                        "duration": "4 hours",
                        "link": f"https://www.linkedin.com/learning/{skill_lower}-essential-training",
                        "educator": "Professional Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.6/5"
                    },
                    {
                        "course_name": f"Introduction to {keyword}",
                        "platform": "edX",
                        "duration": "5 weeks",
                        "link": f"https://www.edx.org/course/introduction-to-{skill_lower}",
                        "educator": "University Partner",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.3/5"
                    },
                    {
                        "course_name": f"{keyword} Professional Certificate",
                        "platform": "Coursera",
                        "duration": "4 months",
                        "link": f"https://www.coursera.org/professional-certificates/{skill_lower}",
                        "educator": "Industry Leader",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.5/5"
                    }
                ])
        
        return recommendations
    

class GPTService:
    """Service class for GPT-based reranking using Azure OpenAI"""
    
    @staticmethod
    async def rerank_jobs(skills: List[str], jobs: List[Dict]) -> List[Dict]:
        """Re-rank jobs using Azure OpenAI GPT"""
        
        if not jobs:
            return []
        
        # Prepare job data for GPT
        processed_jobs = []
        for job in jobs[:25]:  # Limit to top 25 for GPT processing
            processed_job = {
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "keywords": job.get("keywords", "")[:200],
                "description": job.get("description", "")[:300] if job.get("description") else "",
                "similarity": round(job.get("similarity", 0), 3)
            }
            processed_jobs.append(processed_job)
        
        jobs_json = json.dumps(processed_jobs, indent=2)
        skills_str = ', '.join(skills)
        
        prompt = f"""
You are an expert job matcher. Analyze the job seeker's skills and rank the jobs by relevance.

Job Seeker Skills: {skills_str}

Jobs to rank:
{jobs_json}

Instructions:
1. Rank jobs from best to worst match based on skill alignment
2. Assign match_percentage between 100-40 based on how well skills align with job requirements
3. Consider exact skill matches, related skills, and transferable skills
4. Higher percentage for closer skill matches
5. Return ONLY valid JSON array
6. Give me only unique ncspjobid in the json array correctly. 

Required format: [{{"ncspjobid": 123, "title": "Job Title", "match_percentage": 85}}, ...]
"""

        try:
            logger.info("Reranking jobs with Azure GPT...")
            
            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a skilled career advisor. Return only valid JSON array. No explanation text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()
            
            try:
                ranked_jobs = json.loads(content)
                if isinstance(ranked_jobs, list) and len(ranked_jobs) > 0:
                    logger.info(f"Successfully ranked {len(ranked_jobs)} jobs")
                    return ranked_jobs
                else:
                    logger.warning("GPT returned empty or invalid list")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response: {e}")
            
            # Fallback to similarity-based ranking
            logger.info("Using intelligent skill-based fallback ranking")
            fallback_jobs = []

            for job in processed_jobs:
                similarity = job.get("similarity", 0.0)
                job_keywords = job.get("keywords", "").lower()
                job_title = job.get("title", "").lower()
                job_description = job.get("description", "").lower()
                
                # Parse job keywords into list
                job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]
                
                # Combine all job text for matching
                job_text = f"{job_keywords} {job_title} {job_description}"
                
                # Track skill matching details
                skill_matches = 0
                partial_matches = 0
                matched_job_keywords = []
                unmatched_job_keywords = []
                user_skills_matched = []
                total_skills = len(skills)
                
                # Check each job keyword against user skills
                for job_kw in job_keyword_list:
                    job_kw_lower = job_kw.lower().strip()
                    kw_matched = False
                    
                    for skill in skills:
                        skill_lower = skill.lower()
                        
                        # Check if user skill matches this job keyword
                        if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                            matched_job_keywords.append(job_kw)
                            if skill not in user_skills_matched:
                                user_skills_matched.append(f"{skill} (keywords)")
                            kw_matched = True
                            break
                    
                    if not kw_matched:
                        unmatched_job_keywords.append(job_kw)
                
                # Now check user skills for matches in title/description if not found in keywords
                for skill in skills:
                    skill_lower = skill.lower()
                    already_matched = any(skill in matched for matched in user_skills_matched)
                    
                    if not already_matched:
                        # Title match (high weight)
                        if skill_lower in job_title:
                            skill_matches += 0.8
                            user_skills_matched.append(f"{skill} (title)")
                        # Description match (medium weight)
                        elif skill_lower in job_description:
                            skill_matches += 0.6
                            user_skills_matched.append(f"{skill} (description)")
                        # Partial match (low weight)
                        elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                            partial_matches += 0.3
                            user_skills_matched.append(f"{skill} (partial)")
                
                # Calculate keyword match score based on matched job keywords
                keyword_matches_count = len(matched_job_keywords)
                if job_keyword_list:
                    keyword_match_score = keyword_matches_count / len(job_keyword_list)
                else:
                    keyword_match_score = 0
                
                # Calculate user skill match score
                user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0
                
                # Combine both scores (70% user skills, 30% keyword coverage)
                combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)
                
                # Add similarity component (20% weight)
                final_score = (combined_score * 0.8) + (similarity * 0.2)
                
                # Convert to percentage with realistic ranges
                if final_score >= 0.8:
                    match_percentage = 85 + (final_score - 0.8) * 75  # 85-100%
                elif final_score >= 0.6:
                    match_percentage = 70 + (final_score - 0.6) * 75  # 70-85%
                elif final_score >= 0.4:
                    match_percentage = 55 + (final_score - 0.4) * 75  # 55-70%
                elif final_score >= 0.2:
                    match_percentage = 40 + (final_score - 0.2) * 75  # 40-55%
                else:
                    match_percentage = 25 + final_score * 75  # 25-40%
                
                # Cap at reasonable limits
                match_percentage = max(25, min(98, match_percentage))
                
                fallback_jobs.append({
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": round(match_percentage, 1),
                    "keywords_matched": matched_job_keywords,
                    "keywords_unmatched": unmatched_job_keywords,
                    "user_skills_matched": user_skills_matched,
                    "keyword_match_score": round(keyword_match_score, 2),
                    "similarity_used": round(similarity, 3)
                })
            return fallback_jobs
            
        except Exception as e:
            logger.error(f"GPT reranking failed: {e}")
            # Fallback to similarity-based ranking
            logger.info("Using intelligent skill-based fallback ranking")
            fallback_jobs = []

            for job in processed_jobs:
                similarity = job.get("similarity", 0.0)
                job_keywords = job.get("keywords", "").lower()
                job_title = job.get("title", "").lower()
                job_description = job.get("description", "").lower()
                
                # Parse job keywords into list
                job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]
                
                # Combine all job text for matching
                job_text = f"{job_keywords} {job_title} {job_description}"
                
                # Track skill matching details
                skill_matches = 0
                partial_matches = 0
                matched_job_keywords = []
                unmatched_job_keywords = []
                user_skills_matched = []
                total_skills = len(skills)
                
                # Check each job keyword against user skills
                for job_kw in job_keyword_list:
                    job_kw_lower = job_kw.lower().strip()
                    kw_matched = False
                    
                    for skill in skills:
                        skill_lower = skill.lower()
                        
                        # Check if user skill matches this job keyword
                        if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                            matched_job_keywords.append(job_kw)
                            if skill not in user_skills_matched:
                                user_skills_matched.append(f"{skill} (keywords)")
                            kw_matched = True
                            break
                    
                    if not kw_matched:
                        unmatched_job_keywords.append(job_kw)
                
                # Now check user skills for matches in title/description if not found in keywords
                for skill in skills:
                    skill_lower = skill.lower()
                    already_matched = any(skill in matched for matched in user_skills_matched)
                    
                    if not already_matched:
                        # Title match (high weight)
                        if skill_lower in job_title:
                            skill_matches += 0.8
                            user_skills_matched.append(f"{skill} (title)")
                        # Description match (medium weight)
                        elif skill_lower in job_description:
                            skill_matches += 0.6
                            user_skills_matched.append(f"{skill} (description)")
                        # Partial match (low weight)
                        elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                            partial_matches += 0.3
                            user_skills_matched.append(f"{skill} (partial)")
                
                # Calculate keyword match score based on matched job keywords
                keyword_matches_count = len(matched_job_keywords)
                if job_keyword_list:
                    keyword_match_score = keyword_matches_count / len(job_keyword_list)
                else:
                    keyword_match_score = 0
                
                # Calculate user skill match score
                user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0
                
                # Combine both scores (70% user skills, 30% keyword coverage)
                combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)
                
                # Add similarity component (20% weight)
                final_score = (combined_score * 0.8) + (similarity * 0.2)
                
                # Convert to percentage with realistic ranges
                if final_score >= 0.8:
                    match_percentage = 85 + (final_score - 0.8) * 75  # 85-100%
                elif final_score >= 0.6:
                    match_percentage = 70 + (final_score - 0.6) * 75  # 70-85%
                elif final_score >= 0.4:
                    match_percentage = 55 + (final_score - 0.4) * 75  # 55-70%
                elif final_score >= 0.2:
                    match_percentage = 40 + (final_score - 0.2) * 75  # 40-55%
                else:
                    match_percentage = 25 + final_score * 75  # 25-40%
                
                # Cap at reasonable limits
                match_percentage = max(25, min(98, match_percentage))
                
                fallback_jobs.append({
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": round(match_percentage, 1),
                    "keywords_matched": matched_job_keywords,
                    "keywords_unmatched": unmatched_job_keywords,
                    "user_skills_matched": user_skills_matched,
                    "keyword_match_score": round(keyword_match_score, 2),
                    "similarity_used": round(similarity, 3)
                })
            return fallback_jobs

# Initialize services
embedding_service = LocalEmbeddingService()
vector_store = FAISSVectorStore()
gpt_service = GPTService()
course_service = CourseRecommendationService()

cv_processor = CVProcessor(
    model_path="all-MiniLM-L6-v2",
    tesseract_path=r"C:\Users\WK929BY\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"  # Update path as needed
)

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Job Search API starting up...")
    
    # Validate environment variables
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_GPT_DEPLOYMENT",
        "DATABASE_URL"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Load FAISS index
    try:
        await vector_store.load_jobs_from_db()
        logger.info("Job search index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load job search index: {e}")
        raise
    
    logger.info("Job Search API started successfully")
    yield
    
    # Shutdown
    logger.info("Job Search API shutting down...")
    embedding_executor.shutdown(wait=True)


async def get_complete_job_details(job_ids: List[str]) -> List[Dict]:
    """Fetch complete job details from database for given job IDs"""
    if not job_ids:
        return []
    
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            rows = await conn.fetch("""
                SELECT ncspjobid, title, keywords, description, date, organizationid, 
                       organization_name, numberofopenings, industryname, sectorname, 
                       functionalareaname, functionalrolename, aveexp, avewage, 
                       gendercode, highestqualification, statename, districtname
                FROM vacancies_summary
                WHERE ncspjobid = ANY($1)
                ORDER BY ncspjobid;
            """, job_ids)
            
            # Convert to list of dictionaries
            complete_jobs = []
            for row in rows:
                job_dict = {
                    'ncspjobid': row['ncspjobid'],
                    'title': row['title'],
                    'keywords': row['keywords'],
                    'description': row['description'],
                    'date': row['date'].isoformat() if row['date'] else None,
                    'organizationid': row['organizationid'],
                    'organization_name': row['organization_name'],
                    'numberofopenings': row['numberofopenings'],
                    'industryname': row['industryname'],
                    'sectorname': row['sectorname'],
                    'functionalareaname': row['functionalareaname'],
                    'functionalrolename': row['functionalrolename'],
                    'aveexp': float(row['aveexp']) if row['aveexp'] else None,
                    'avewage': float(row['avewage']) if row['avewage'] else None,
                    'gendercode': row['gendercode'],
                    'highestqualification': row['highestqualification'],
                    'statename': row['statename'],
                    'districtname': row['districtname']
                }
                complete_jobs.append(job_dict)
            
            return complete_jobs
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"Failed to fetch complete job details: {e}")
        return []

app = FastAPI(
    title="Job Search API",
    description="AI-powered job search using skills matching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )


# Main job search endpoint
@app.post("/search_jobs", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """
    Search for relevant job postings based on skills
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Job search request: {len(request.skills)} skills, limit: {request.limit}")
        
        # Combine skills into text for embedding
        skills_text = " ".join(request.skills)
        
        # Generate embedding for skills
        skills_embedding = await embedding_service.get_embedding(skills_text)
        logger.info(f"Generated embedding for skills: {skills_text[:100]}...")
        
        # Search similar jobs using FAISS
        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=50)
        
        if not similar_jobs:
            return JobSearchResponse(
                jobs=[],
                query_skills=request.skills,
                total_found=0,
                processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000)
            )
        
        # Re-rank with Azure GPT
        ranked_jobs = await gpt_service.rerank_jobs(request.skills, similar_jobs)

        ranked_jobs.sort(key=lambda job: job.get("match_percentage", 0), reverse=True)
        
        job_ids = [job["ncspjobid"] for job in ranked_jobs[:request.limit]]
        complete_jobs = await get_complete_job_details(job_ids)
        # Convert to response format
        job_results = []
        for job_data in ranked_jobs[:request.limit]:
            # Find complete job data from database
            complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
            
            job_result = JobResult(
                ncspjobid=job_data["ncspjobid"],
                title=job_data["title"],
                match_percentage=job_data["match_percentage"],
                similarity_score=next((j.get("similarity") for j in similar_jobs if j["ncspjobid"] == job_data["ncspjobid"]), None),
                keywords=complete_job.get("keywords"),
                description=complete_job.get("description"),
                date=complete_job.get("date"),
                organizationid=complete_job.get("organizationid"),
                organization_name=complete_job.get("organization_name"),
                numberofopenings=complete_job.get("numberofopenings"),
                industryname=complete_job.get("industryname"),
                sectorname=complete_job.get("sectorname"),
                functionalareaname=complete_job.get("functionalareaname"),
                functionalrolename=complete_job.get("functionalrolename"),
                aveexp=complete_job.get("aveexp"),
                avewage=complete_job.get("avewage"),
                gendercode=complete_job.get("gendercode"),
                highestqualification=complete_job.get("highestqualification"),
                statename=complete_job.get("statename"),
                districtname=complete_job.get("districtname"),
                keywords_matched=job_data.get("keywords_matched"),
                keywords_unmatched=job_data.get("keywords_unmatched"),
                user_skills_matched=job_data.get("user_skills_matched"),
                keyword_match_score=job_data.get("keyword_matches")
            )
            job_results.append(job_result)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        logger.info(f"Job search completed: {len(job_results)} jobs returned in {processing_time_ms}ms")
        
        return JobSearchResponse(
            jobs=job_results,
            query_skills=request.skills,
            total_found=len(ranked_jobs),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job search failed: {e}")
        raise HTTPException(status_code=500, detail="Job search failed")
    
@app.post("/recommend_courses", response_model=CourseRecommendationResponse)
async def recommend_courses(request: CourseRecommendationRequest) -> CourseRecommendationResponse:
    """
    Get course recommendations for unmatched keywords
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Course recommendation request for {len(request.keywords_unmatched)} keywords: {request.keywords_unmatched}")
        
        # Get course recommendations from GPT
        recommendations_data = await course_service.get_course_recommendations(request.keywords_unmatched)
        
        # Convert to response format
        course_recommendations = []
        for course_data in recommendations_data:
            try:
                course_rec = CourseRecommendation(
                    course_name=course_data.get("course_name", "Unknown Course"),
                    platform=course_data.get("platform", "Unknown Platform"),
                    duration=course_data.get("duration", "Unknown Duration"),
                    link=course_data.get("link", ""),
                    educator=course_data.get("educator", "Unknown Educator"),
                    skill_covered=course_data.get("skill_covered", ""),
                    difficulty_level=course_data.get("difficulty_level"),
                    rating=course_data.get("rating")
                )
                course_recommendations.append(course_rec)
            except Exception as e:
                logger.error(f"Error processing course recommendation: {e}")
                continue
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        logger.info(f"Course recommendations completed: {len(course_recommendations)} courses returned in {processing_time_ms}ms")
        
        return CourseRecommendationResponse(
            recommendations=course_recommendations,
            keywords_processed=request.keywords_unmatched,
            total_recommendations=len(course_recommendations),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Course recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Course recommendation failed")


#New Endpoint added

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle chat messages"""
    try:
        logger.info(f"Chat request: {request.message[:50]}...")
        response = await chat_service.handle_chat_message(request)
        return response
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return ChatResponse(
            response="I'm having trouble understanding. Could you tell me about your technical skills?",
            message_type="text",
            chat_phase="profile_building"
        )

@app.post("/upload_cv", response_model=CVAnalysisResponse)
async def upload_cv_enhanced(cv_file: UploadFile = File(...)) -> CVAnalysisResponse:
    """Enhanced CV upload with complete analysis and job matching"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.jpeg'}
        file_ext = '.' + cv_file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Processing CV upload: {cv_file.filename}")
        
        # Read file content
        file_content = await cv_file.read()
        
        # Process CV using enhanced processor
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Get job search text
        search_text = cv_processor.get_job_search_text(cv_profile)
        
        jobs_found = []
        total_jobs = 0
        
        # Perform job search if we have meaningful skills
        if cv_profile.skills and len(cv_profile.skills) >= 2:
            try:
                # Generate embedding for combined profile text
                profile_embedding = await embedding_service.get_embedding(search_text)
                
                # Search similar jobs
                similar_jobs = await vector_store.search_similar_jobs(
                    profile_embedding, 
                    top_k=30
                )
                
                if similar_jobs:
                    # Re-rank jobs using GPT with extracted skills
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    
                    # Get complete job details for top matches
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[:10]]
                    complete_jobs = await get_complete_job_details(job_ids)
                    
                    # Format job results
                    for job_data in ranked_jobs[:10]:
                        complete_job = next(
                            (j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), 
                            {}
                        )
                        
                        if complete_job:
                            jobs_found.append({
                                "ncspjobid": job_data["ncspjobid"],
                                "title": job_data["title"],
                                "organization_name": complete_job.get("organization_name", ""),
                                "match_percentage": job_data["match_percentage"],
                                "statename": complete_job.get("statename", ""),
                                "districtname": complete_job.get("districtname", ""),
                                "avewage": complete_job.get("avewage", 0),
                                "aveexp": complete_job.get("aveexp", 0),
                                "keywords": complete_job.get("keywords", ""),
                                "functionalrolename": complete_job.get("functionalrolename", ""),
                                "industryname": complete_job.get("industryname", ""),
                                "skills_matched": job_data.get("keywords_matched", []),
                                "similarity_score": job_data.get("similarity_used", 0)
                            })
                    
                    total_jobs = len(ranked_jobs)
                    
            except Exception as job_search_error:
                logger.error(f"Job search failed during CV processing: {job_search_error}")
                # Continue without job results
        
        # Generate processing recommendations
        recommendations = _generate_cv_recommendations(cv_profile)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Success response
        success_message = f"✅ Successfully processed your CV! "
        
        if jobs_found:
            success_message += f"Found {len(jobs_found)} matching jobs with {len(cv_profile.skills)} extracted skills."
        else:
            success_message += f"Extracted {len(cv_profile.skills)} skills. Try refining your CV for better job matches."
        
        logger.info(f"CV processing completed: {cv_profile.confidence_score} confidence, "
                   f"{len(jobs_found)} jobs, {processing_time_ms}ms")
        
        return CVAnalysisResponse(
            success=True,
            message=success_message,
            profile=cv_processor.to_dict(cv_profile),
            jobs=jobs_found,
            total_jobs_found=total_jobs,
            processing_time_ms=processing_time_ms,
            confidence_score=cv_profile.confidence_score,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV upload processing failed: {e}")
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return CVAnalysisResponse(
            success=False,
            message=f"Failed to process CV: {str(e)}",
            processing_time_ms=processing_time_ms
        )

def _generate_cv_recommendations(profile: CVProfile) -> List[str]:
    """Generate recommendations based on CV analysis"""
    recommendations = []
    
    # Skills recommendations
    if len(profile.skills) < 5:
        recommendations.append("💡 Add more technical skills to improve job matching")
    
    # Contact information
    if not profile.email:
        recommendations.append("📧 Add contact email for better profile completeness")
    
    if not profile.phone:
        recommendations.append("📱 Include phone number in your CV")
    
    # Experience recommendations
    if len(profile.experience) == 0:
        recommendations.append("💼 Add work experience details for better job matching")
    elif len(profile.experience) < 2:
        recommendations.append("💼 Include more work experience entries if available")
    
    # Education recommendations
    if len(profile.education) == 0:
        recommendations.append("🎓 Add education background to strengthen your profile")
    
    # Summary recommendations
    if not profile.summary:
        recommendations.append("📝 Add a professional summary to highlight your strengths")
    
    # Confidence-based recommendations
    if profile.confidence_score < 0.5:
        recommendations.append("⚡ Consider adding more detailed information to improve CV quality")
    elif profile.confidence_score < 0.7:
        recommendations.append("📈 Good CV structure! Add a few more details for optimal results")
    
    # Skills gap analysis
    high_demand_skills = [
        "Python", "JavaScript", "React", "Node.js", "AWS", "Docker", 
        "Kubernetes", "SQL", "MongoDB", "Git", "CI/CD"
    ]
    
    missing_skills = [skill for skill in high_demand_skills 
                     if skill.lower() not in [s.lower() for s in profile.skills]]
    
    if missing_skills:
        recommendations.append(f"🚀 Consider learning in-demand skills: {', '.join(missing_skills[:3])}")
    
    return recommendations[:5]  # Limit to top 5 recommendations

# Add this new endpoint for CV analysis without job search
@app.post("/analyze_cv", response_model=CVAnalysisResponse)
async def analyze_cv_only(cv_file: UploadFile = File(...)) -> CVAnalysisResponse:
    """Analyze CV structure and extract data without job matching"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.jpeg'}
        file_ext = '.' + cv_file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Analyzing CV: {cv_file.filename}")
        
        # Read and process CV
        file_content = await cv_file.read()
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Generate recommendations
        recommendations = _generate_cv_recommendations(cv_profile)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        analysis_message = f"📊 CV Analysis Complete! Extracted {len(cv_profile.skills)} skills, "
        analysis_message += f"{len(cv_profile.experience)} experience entries, "
        analysis_message += f"confidence score: {cv_profile.confidence_score:.1%}"
        
        return CVAnalysisResponse(
            success=True,
            message=analysis_message,
            profile=cv_processor.to_dict(cv_profile),
            processing_time_ms=processing_time_ms,
            confidence_score=cv_profile.confidence_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"CV analysis failed: {e}")
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return CVAnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            processing_time_ms=processing_time_ms
        )

@app.post("/upload_cv_chat", response_model=ChatResponse)
async def upload_cv_for_chat(cv_file: UploadFile = File(...)) -> ChatResponse:
    """Upload CV and get chat-style response with job matches"""
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Process CV using the enhanced processor
        file_content = await cv_file.read()
        cv_profile = await cv_processor.process_cv(file_content, cv_file.filename)
        
        # Generate chat response with job matches using CV service
        chat_response = await cv_chat_service.handle_cv_upload_chat(cv_profile)
        
        return chat_response
        
    except Exception as e:
        logger.error(f"CV chat upload failed: {e}")
        return ChatResponse(
            response="I had trouble processing your CV. Let's build your profile by chatting about your skills!",
            message_type="text",
            chat_phase="profile_building"
        )

@app.post("/chat_with_cv", response_model=ChatResponse)
async def chat_with_cv_context(request: ChatWithCVRequest) -> ChatResponse:
    """Handle chat with CV context for follow-up questions"""
    try:
        if request.cv_profile_data:
            # Convert dict back to CVProfile if needed
            cv_profile = CVProfile(**request.cv_profile_data)
            
            # Create a ChatRequest for the CV service
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )
            
            response = await cv_chat_service.handle_cv_followup_chat(chat_request, cv_profile)
        else:
            # Fall back to regular chat
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )
            response = await simple_chat_service.handle_chat_message(chat_request)
        
        return response
    except Exception as e:
        logger.error(f"Chat with CV context failed: {e}")
        return ChatResponse(
            response="I can help you find jobs. What would you like to do?",
            message_type="text",
            chat_phase="job_searching"
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
        log_level="info"
    )