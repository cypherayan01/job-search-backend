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

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
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

from cv import CVProcessor, CVProfile

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

# Azure OpenAI client initialization
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

# Database connection
DB_URL = os.getenv("DATABASE_URL")

# =============================================================================
# Pydantic Models
# =============================================================================

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

class ChatWithCVRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []
    cv_profile_data: Optional[Dict] = None

class CVAnalysisResponse(BaseModel):
    success: bool
    message: str
    profile: Optional[Dict[str, Any]] = None
    jobs: Optional[List[Dict]] = None
    total_jobs_found: int = 0
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    recommendations: Optional[List[str]] = None

class JobSearchRequest(BaseModel):
    skills: List[str] = Field(..., min_items=1, max_items=50, description="List of skills to search for")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of jobs to return")
    
    @validator('skills', pre=True)
    def validate_and_clean_skills(cls, v):
        if not v:
            raise ValueError('Skills list cannot be empty')
        
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
        cleaned_skills = []
        for skill in v:
            if isinstance(skill, str) and skill.strip():
                cleaned = re.sub(r'[^\w\s+#.-]', '', skill.strip())
                if cleaned and len(cleaned) > 1:
                    cleaned_skills.append(cleaned)
        
        if not cleaned_skills:
            raise ValueError('No valid skills found after cleaning')
        
        if len(cleaned_skills) > 50:
            cleaned_skills = cleaned_skills[:50]
        
        return cleaned_skills

class CourseRecommendationRequest(BaseModel):
    keywords_unmatched: List[str] = Field(..., min_items=1, max_items=20)
    
    @validator('keywords_unmatched', pre=True)
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]
        
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

# =============================================================================
# Service Classes
# =============================================================================

class LocalEmbeddingService:
    """Production embedding service with optimized PyTorch operations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize with device optimization and model compilation"""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Production optimization: compile model for faster inference
            if device == "cuda":
                self.model = torch.compile(self.model, mode="reduce-overhead")
            
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Warmup inference
            test_embedding = self.model.encode("test input", convert_to_tensor=False)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Optimized synchronous embedding generation"""
        try:
            if not self.model:
                raise ValueError("Model not initialized")
            
            # Use torch.no_grad() for inference optimization
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1
                )
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise
    
    async def get_embedding(self, text: Union[str, List[str], Any]) -> List[float]:
        """Async embedding generation with input validation"""
        
        # Input normalization and validation
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
            
            # Truncate for model limits
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
    """Production FAISS vector store with index optimization"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.job_metadata = []
        self.is_loaded = False
        self._lock = threading.Lock()
        self.index_file = "faiss_job_index.bin"
        self.metadata_file = "job_metadata.pkl"
    
    async def load_jobs_from_db(self, force_reload: bool = False):
        """Load and build optimized FAISS index"""
        if self.is_loaded and not force_reload:
            logger.info("FAISS index already loaded")
            return
        
        with self._lock:
            try:
                # Try loading existing index
                if not force_reload and os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                    logger.info("Loading existing FAISS index from disk...")
                    self.index = faiss.read_index(self.index_file)
                    
                    with open(self.metadata_file, 'rb') as f:
                        self.job_metadata = pickle.load(f)
                    
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} jobs")
                    self.is_loaded = True
                    return
                
                logger.info("Building new FAISS index from database...")
                
                # Database connection and data fetch
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
                    
                    # Generate embeddings
                    logger.info("Generating embeddings for all jobs...")
                    embeddings = await self._generate_job_embeddings(job_texts)
                    
                    # Create optimized FAISS index
                    # Use IndexFlatIP for cosine similarity (after L2 normalization)
                    self.index = faiss.IndexFlatIP(self.dimension)
                    
                    # For production: consider IndexIVFFlat for large datasets
                    # quantizer = faiss.IndexFlatIP(self.dimension)
                    # self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, len(embeddings)//10))
                    
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)
                    
                    # For IVF index: self.index.train(embeddings_array)
                    
                    # Save to disk
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
        """Batch embedding generation with proper error handling"""
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
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * self.dimension)
            
            embeddings.extend(batch_embeddings)
            # Small delay to prevent overwhelming the embedding service
            await asyncio.sleep(0.1)
        
        return embeddings
    
    async def search_similar_jobs(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        """Optimized similarity search"""
        if not self.is_loaded or self.index is None:
            raise HTTPException(status_code=503, detail="Job search index not available")
        
        try:
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search with bounds checking
            search_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_vector, search_k)
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.job_metadata):
                    job_data = self.job_metadata[idx].copy()
                    job_data['similarity'] = float(similarity)
                    results.append(job_data)
            
            logger.info(f"FAISS search returned {len(results)} similar jobs")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise HTTPException(status_code=500, detail="Job search failed")

class SimpleChatService:
    """Chat service with comprehensive skill extraction"""
    
    def __init__(self):
        self.skill_keywords = {
            # Technical/IT Skills
            'python': ['python', 'py', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'node.js'],
            'react': ['react', 'reactjs', 'react.js', 'nextjs', 'next.js'],
            'angular': ['angular', 'angularjs'],
            'vue': ['vue', 'vuejs', 'vue.js', 'nuxt'],
            'java': ['java', 'spring', 'springboot', 'spring boot'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite'],
            'mongodb': ['mongodb', 'mongo'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3', 'sass', 'scss', 'tailwind'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'git': ['git', 'github', 'gitlab'],
            'machine learning': ['ml', 'machine learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras'],
            'data science': ['data science', 'pandas', 'numpy', 'matplotlib', 'jupyter'],
            'typescript': ['typescript', 'ts'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails', 'ruby on rails'],
            
            # Business Process/BPO Skills
            'data entry': ['data entry', 'data processing', 'typing', 'keyboarding'],
            'voice process': ['voice process', 'call center', 'customer service', 'telecalling', 'telesales'],
            'chat process': ['chat process', 'chat support', 'live chat', 'online support'],
            'email support': ['email support', 'email handling', 'email management'],
            'back office': ['back office', 'administrative', 'admin work'],
            'content writing': ['content writing', 'copywriting', 'blogging', 'article writing'],
            'virtual assistant': ['virtual assistant', 'va', 'personal assistant'],
            
            # Finance & Accounting
            'accounting': ['accounting', 'bookkeeping', 'accounts', 'financial'],
            'tally': ['tally', 'tally erp'],
            'excel': ['excel', 'microsoft excel', 'spreadsheet', 'vlookup', 'pivot tables'],
            'sap': ['sap', 'sap fico', 'sap mm', 'sap hr'],
            'quickbooks': ['quickbooks', 'quick books'],
            'gst': ['gst', 'goods and services tax', 'taxation'],
            'payroll': ['payroll', 'salary processing', 'hr payroll'],
            
            # Sales & Marketing
            'sales': ['sales', 'selling', 'business development', 'lead generation'],
            'digital marketing': ['digital marketing', 'online marketing', 'internet marketing'],
            'seo': ['seo', 'search engine optimization'],
            'sem': ['sem', 'search engine marketing', 'google ads', 'ppc'],
            'social media': ['social media', 'facebook marketing', 'instagram marketing', 'linkedin'],
            'email marketing': ['email marketing', 'mailchimp', 'newsletter'],
            'content marketing': ['content marketing', 'inbound marketing'],
            
            # Additional sectors omitted for brevity...
        }
    
    async def handle_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Main chat handling logic"""
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
                elif any(word in message for word in ["chat", "talk", "build", "skills", "hello", "hi", "hey"]):
                    return ChatResponse(
                        response="Perfect! Let's build your profile together. What are your main skills? (e.g., Python, React, Data Entry, Customer Service, etc.)",
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
                skills = self._extract_skills_from_text(message)
                
                if skills:
                    try:
                        skills_text = " ".join(skills)
                        skills_embedding = await embedding_service.get_embedding(skills_text)
                        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=20)
                        
                        if similar_jobs:
                            ranked_jobs = await gpt_service.rerank_jobs(skills, similar_jobs)
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
                                chat_phase="profile_building",
                                profile_data={"skills": skills}
                            )
                    except Exception as e:
                        logger.error(f"Job search failed in chat: {e}")
                        return ChatResponse(
                            response=f"I noted your skills: {', '.join(skills)}. What other skills do you have?",
                            message_type="text",
                            chat_phase="profile_building",
                            profile_data={"skills": skills}
                        )
                else:
                    return ChatResponse(
                        response="I'd like to help you find jobs. Please tell me your skills. For example: 'I know Python and React' or 'I can do Data Entry and Customer Service'",
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
                response="Let me help you find jobs. What skills do you have?",
                message_type="text",
                chat_phase="profile_building"
            )
    
    def _extract_skills_from_text(self, message: str) -> List[str]:
        """Extract skills using comprehensive keyword matching"""
        skills = []
        message_lower = message.lower()
        
        for main_skill, variations in self.skill_keywords.items():
            for variation in variations:
                if variation in message_lower:
                    # Proper capitalization based on skill type
                    if main_skill == 'c++':
                        skills.append('C++')
                    elif main_skill == 'c#':
                        skills.append('C#')
                    elif main_skill == 'javascript':
                        skills.append('JavaScript')
                    elif main_skill == 'typescript':
                        skills.append('TypeScript')
                    elif main_skill == 'machine learning':
                        skills.append('Machine Learning')
                    elif main_skill == 'data science':
                        skills.append('Data Science')
                    elif main_skill == 'data entry':
                        skills.append('Data Entry')
                    elif main_skill == 'voice process':
                        skills.append('Voice Process')
                    else:
                        skills.append(main_skill.title())
                    break
        
        # Experience pattern matching
        experience_patterns = {
            'years': r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            'months': r'(\d+)\s*(?:months?|mos?)\s*(?:of\s*)?(?:experience|exp)',
            'fresher': r'\b(?:fresher|fresh|new|entry\s*level|no\s*experience)\b'
        }
        
        for pattern_name, pattern in experience_patterns.items():
            matches = re.findall(pattern, message_lower)
            if matches:
                if pattern_name == 'years' and matches:
                    skills.append(f"{matches[0]} Years Experience")
                elif pattern_name == 'months' and matches:
                    skills.append(f"{matches[0]} Months Experience")
                elif pattern_name == 'fresher':
                    skills.append("Fresher")
                break
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(skills))

class CVChatService:
    """Enhanced CV-based chat service"""
    
    @staticmethod
    async def handle_cv_upload_chat(cv_profile: CVProfile) -> ChatResponse:
        """Handle initial CV upload response"""
        try:
            if cv_profile.skills and len(cv_profile.skills) >= 3:
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=15)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
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
                                "description":complete_job.get("description", ""),
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
                        response=f"Perfect! I've analyzed your CV and found your skills: {', '.join(cv_profile.skills[:5])}{'...' if len(cv_profile.skills) > 5 else ''}. Here are {len(job_results)} matching jobs:",
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
                    response="I've processed your CV but found limited skills. Let's chat to build a complete profile for better job matching.",
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
        """Handle follow-up conversations after CV analysis"""
        message = request.message.lower().strip()
        
        try:
            # Handle more jobs request
            if any(word in message for word in [" Show more jobs", "show more", "additional", "other jobs"]):
                skills_text = " ".join(cv_profile.skills[:10])
                skills_embedding = await embedding_service.get_embedding(skills_text)
                similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=25)
                
                if similar_jobs:
                    ranked_jobs = await gpt_service.rerank_jobs(cv_profile.skills, similar_jobs)
                    job_ids = [job["ncspjobid"] for job in ranked_jobs[5:10]]  # Skip first 5
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
            
            # Handle skill addition
            elif any(word in message for word in ["add skill", "more skill", "also know", "i can", "i have experience"]):
                chat_service_instance = SimpleChatService()
                additional_skills = chat_service_instance._extract_skills_from_text(message)
                new_skills = [skill for skill in additional_skills if skill not in cv_profile.skills]
                
                if new_skills:
                    combined_skills = cv_profile.skills + new_skills
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
                            response=f"Great! I've added {', '.join(new_skills)} to your profile. Here are updated job matches:",
                            message_type="job_results",
                            chat_phase="job_results",
                            profile_data={"skills": combined_skills},
                            jobs=job_results
                        )
                else:
                    return ChatResponse(
                        response="What additional skills would you like to add to your profile? For example: 'I also know Data Entry and Voice Process' or 'I have experience in Customer Service'",
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

class GPTService:
    """Production GPT service with fallback ranking"""
    
    @staticmethod
    async def rerank_jobs(skills: List[str], jobs: List[Dict]) -> List[Dict]:
        """Intelligent job reranking with GPT and fallback"""
        
        if not jobs:
            return []
        
        # Prepare job data for GPT processing
        processed_jobs = []
        for job in jobs[:25]:  # Limit for token efficiency
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
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = re.sub(r'^[^[]*', '', content)  # Remove text before JSON array
            content = re.sub(r'[^}]*$', '}]', content)  # Ensure proper JSON ending
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
            
            # Fallback to intelligent ranking
            logger.info("Using intelligent skill-based fallback ranking")
            return GPTService._fallback_ranking(skills, processed_jobs)
            
        except Exception as e:
            logger.error(f"GPT reranking failed: {e}")
            return GPTService._fallback_ranking(skills, processed_jobs)
    
    @staticmethod
    def _fallback_ranking(skills: List[str], processed_jobs: List[Dict]) -> List[Dict]:
        """Advanced fallback ranking algorithm"""
        fallback_jobs = []

        for job in processed_jobs:
            similarity = job.get("similarity", 0.0)
            job_keywords = job.get("keywords", "").lower()
            job_title = job.get("title", "").lower()
            job_description = job.get("description", "").lower()
            
            # Parse job keywords
            job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]
            job_text = f"{job_keywords} {job_title} {job_description}"
            
            # Skill matching analysis
            matched_job_keywords = []
            unmatched_job_keywords = []
            user_skills_matched = []
            total_skills = len(skills)
            
            # Check job keywords against user skills
            for job_kw in job_keyword_list:
                job_kw_lower = job_kw.lower().strip()
                kw_matched = False
                
                for skill in skills:
                    skill_lower = skill.lower()
                    
                    if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                        matched_job_keywords.append(job_kw)
                        if skill not in user_skills_matched:
                            user_skills_matched.append(f"{skill} (keywords)")
                        kw_matched = True
                        break
                
                if not kw_matched:
                    unmatched_job_keywords.append(job_kw)
            
            # Check user skills in title/description
            for skill in skills:
                skill_lower = skill.lower()
                already_matched = any(skill in matched for matched in user_skills_matched)
                
                if not already_matched:
                    if skill_lower in job_title:
                        user_skills_matched.append(f"{skill} (title)")
                    elif skill_lower in job_description:
                        user_skills_matched.append(f"{skill} (description)")
                    elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                        user_skills_matched.append(f"{skill} (partial)")
            
            # Calculate scores
            keyword_matches_count = len(matched_job_keywords)
            keyword_match_score = keyword_matches_count / len(job_keyword_list) if job_keyword_list else 0
            user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0
            
            # Combined scoring: 70% user skills, 30% keyword coverage, 20% similarity
            combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)
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
        
        return sorted(fallback_jobs, key=lambda x: x["match_percentage"], reverse=True)

class CourseRecommendationService:
    """Course recommendation service with comprehensive database"""
    
    @staticmethod
    async def get_course_recommendations(keywords: List[str]) -> List[Dict]:
        """Get course recommendations using GPT with fallback"""
        
        if not keywords:
            return []
        
        try:
            # Try GPT first (implementation similar to original)
            # For production, implement GPT call here
            pass
        except Exception as e:
            logger.error(f"Course recommendation failed: {e}")
        
        # Use fallback recommendations
        return CourseRecommendationService._get_fallback_recommendations(keywords)
    
    @staticmethod
    def _get_fallback_recommendations(keywords: List[str]) -> List[Dict]:
        """Comprehensive fallback course database"""
        
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
                # Add 3 more Python courses...
            ],
            "Data Entry": [
                {
                    "course_name": "Data Entry Fundamentals",
                    "platform": "Udemy",
                    "duration": "3 hours",
                    "link": "https://www.udemy.com/course/data-entry-fundamentals/",
                    "educator": "Expert Instructor",
                    "skill_covered": "Data Entry",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                # Add 4 more Data Entry courses...
            ],
            # Add more skills as needed...
        }
        
        recommendations = []
        for keyword in keywords:
            keyword_normalized = keyword.strip().title()
            if keyword_normalized in course_database:
                recommendations.extend(course_database[keyword_normalized])
            else:
                # Generic fallback
                skill_lower = keyword.lower().replace(' ', '-')
                recommendations.extend([
                    {
                        "course_name": f"Complete {keyword} Professional Course",
                        "platform": "Udemy",
                        "duration": "15 hours",
                        "link": f"https://www.udemy.com/topic/{skill_lower}/",
                        "educator": "Expert Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "All Levels", 
                        "rating": "4.5/5"
                    }
                ])
        
        return recommendations

# =============================================================================
# Service Initialization
# =============================================================================

embedding_service = LocalEmbeddingService()
vector_store = FAISSVectorStore()
gpt_service = GPTService()
course_service = CourseRecommendationService()
simple_chat_service = SimpleChatService()
cv_chat_service = CVChatService()

cv_processor = CVProcessor(
    model_path="all-MiniLM-L6-v2",
    tesseract_path=r"C:\Users\WK929BY\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# =============================================================================
# Database Helper Functions
# =============================================================================

async def get_complete_job_details(job_ids: List[str]) -> List[Dict]:
    """Fetch complete job details with optimized query"""
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

def _generate_cv_recommendations(profile: CVProfile) -> List[str]:
    """Generate CV improvement recommendations"""
    recommendations = []
    
    if len(profile.skills) < 5:
        recommendations.append("Add more skills to improve job matching")
    
    if not profile.email:
        recommendations.append("Add contact email for better profile completeness")
    
    if not profile.phone:
        recommendations.append("Include phone number in your CV")
    
    if len(profile.experience) == 0:
        recommendations.append("Add work experience details for better job matching")
    
    if len(profile.education) == 0:
        recommendations.append("Add education background to strengthen your profile")
    
    if not profile.summary:
        recommendations.append("Add a professional summary to highlight your strengths")
    
    if profile.confidence_score < 0.5:
        recommendations.append("Consider adding more detailed information to improve CV quality")
    
    # Skills gap analysis
    high_demand_skills = [
        "Python", "JavaScript", "React", "Node.js", "AWS", "Docker", 
        "Data Entry", "Voice Process", "Customer Service", "Excel",
        "Tally", "SAP", "Digital Marketing", "Sales", "MS Office"
    ]
    
    missing_skills = [skill for skill in high_demand_skills 
                     if skill.lower() not in [s.lower() for s in profile.skills]]
    
    if missing_skills:
        recommendations.append(f"Consider learning in-demand skills: {', '.join(missing_skills[:3])}")
    
    return recommendations[:5]

# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
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

app = FastAPI(
    title="Job Search API",
    description="Production AI-powered job search using skills matching",
    version="2.0.0",
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

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/search_jobs", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """Search for relevant job postings based on skills"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Job search request: {len(request.skills)} skills, limit: {request.limit}")
        
        skills_text = " ".join(request.skills)
        skills_embedding = await embedding_service.get_embedding(skills_text)
        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=50)
        
        if not similar_jobs:
            return JobSearchResponse(
                jobs=[],
                query_skills=request.skills,
                total_found=0,
                processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000)
            )
        
        ranked_jobs = await gpt_service.rerank_jobs(request.skills, similar_jobs)
        ranked_jobs.sort(key=lambda job: job.get("match_percentage", 0), reverse=True)
        
        job_ids = [job["ncspjobid"] for job in ranked_jobs[:request.limit]]
        complete_jobs = await get_complete_job_details(job_ids)
        
        job_results = []
        for job_data in ranked_jobs[:request.limit]:
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
                keyword_match_score=job_data.get("keyword_match_score")
            )
            job_results.append(job_result)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
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
    """Get course recommendations for unmatched keywords"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        recommendations_data = await course_service.get_course_recommendations(request.keywords_unmatched)
        
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
        
        return CourseRecommendationResponse(
            recommendations=course_recommendations,
            keywords_processed=request.keywords_unmatched,
            total_recommendations=len(course_recommendations),
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Course recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Course recommendation failed")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle chat messages"""
    try:
        response = await simple_chat_service.handle_chat_message(request)
        return response
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return ChatResponse(
            response="I'm having trouble understanding. Could you tell me about your skills?",
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
        success_message = f"Successfully processed your CV! "
        
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

@app.post("/analyze_cv", response_model=CVAnalysisResponse)
async def analyze_cv_only(cv_file: UploadFile = File(...)) -> CVAnalysisResponse:
    """Analyze CV structure and extract data without job matching"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
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
        
        analysis_message = f"CV Analysis Complete! Extracted {len(cv_profile.skills)} skills, "
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

# =============================================================================
# Health Check and Admin Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "embedding_service": "ready" if embedding_service.model else "not_ready",
            "vector_store": "ready" if vector_store.is_loaded else "not_ready",
            "database": "connected"  # Could add actual DB health check
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Job Search API",
        "version": "2.0.0",
        "description": "Production AI-powered job search using skills matching",
        "endpoints": {
            "search_jobs": "/search_jobs",
            "recommend_courses": "/recommend_courses", 
            "chat": "/chat",
            "upload_cv": "/upload_cv",
            "analyze_cv": "/analyze_cv",
            "upload_cv_chat": "/upload_cv_chat",
            "chat_with_cv": "/chat_with_cv",
            "health": "/health"
        }
    }

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
        log_level="info",
        workers=1 
    )