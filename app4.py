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

class JobResult(BaseModel):
    ncspjobid: str
    title: str
    match_percentage: float = Field(..., ge=0, le=100)
    similarity_score: Optional[float] = None
    keywords: Optional[str] = None
    description: Optional[str] = None

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
            logger.info("Using similarity-based fallback ranking")
            fallback_jobs = []
            for job in processed_jobs:
                similarity = job.get("similarity", 0.0)
                job_keywords = job.get("keywords", "").lower()
                job_title = job.get("title", "").lower()
                job_description = job.get("description", "").lower()
                
                # Combine all job text for matching
                job_text = f"{job_keywords} {job_title} {job_description}"
                
                # Track skill matching details
                skill_matches = 0
                partial_matches = 0
                matched_skills = []
                unmatched_skills = []
                total_skills = len(skills)
                
                for skill in skills:
                    skill_lower = skill.lower()
                    skill_matched = False
                    
                    # Exact keyword match (highest weight)
                    if skill_lower in job_keywords:
                        skill_matches += 1.0
                        matched_skills.append(f"{skill} (keywords)")
                        skill_matched = True
                    # Title match (high weight)
                    elif skill_lower in job_title:
                        skill_matches += 0.8
                        matched_skills.append(f"{skill} (title)")
                        skill_matched = True
                    # Description match (medium weight)
                    elif skill_lower in job_description:
                        skill_matches += 0.6
                        matched_skills.append(f"{skill} (description)")
                        skill_matched = True
                    # Partial match (low weight)
                    elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                        partial_matches += 0.3
                        matched_skills.append(f"{skill} (partial)")
                        skill_matched = True
                    
                    # Track unmatched skills
                    if not skill_matched:
                        unmatched_skills.append(skill)
                
                # Calculate final match percentage
                keyword_score = (skill_matches + partial_matches) / total_skills
                
                # Combine similarity and keyword matching (60% keywords, 40% similarity)
                combined_score = (keyword_score * 0.6) + (similarity * 0.4)
                
                # Convert to percentage with realistic ranges
                if combined_score >= 0.8:
                    match_percentage = 85 + (combined_score - 0.8) * 75  # 85-100%
                elif combined_score >= 0.6:
                    match_percentage = 70 + (combined_score - 0.6) * 75  # 70-85%
                elif combined_score >= 0.4:
                    match_percentage = 55 + (combined_score - 0.4) * 75  # 55-70%
                elif combined_score >= 0.2:
                    match_percentage = 40 + (combined_score - 0.2) * 75  # 40-55%
                else:
                    match_percentage = 25 + combined_score * 75  # 25-40%
                
                # Cap at reasonable limits
                match_percentage = max(25, min(98, match_percentage))
                
                fallback_jobs.append({
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": round(match_percentage, 1),
                    "keyword_matches": round(skill_matches + partial_matches, 2),
                    "keywords_matched": matched_skills,
                    "keywords_unmatched": unmatched_skills,
                    "similarity_used": round(similarity, 3)
                })
            return fallback_jobs
            
        except Exception as e:
            logger.error(f"GPT reranking failed: {e}")
            # Fallback to similarity-based ranking
            return [
                {
                    "ncspjobid": job["ncspjobid"],
                    "title": job["title"],
                    "match_percentage": min(95, max(100, job.get("similarity", 0.5) * 100))
                }
                for job in processed_jobs
            ]

# Initialize services
embedding_service = LocalEmbeddingService()
vector_store = FAISSVectorStore()
gpt_service = GPTService()

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
        
        # Convert to response format
        job_results = []
        for job_data in ranked_jobs[:request.limit]:
            # Find original job data for additional info
            original_job = next((j for j in similar_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})
            
            job_result = JobResult(
                ncspjobid=job_data["ncspjobid"],
                title=job_data["title"],
                match_percentage=job_data["match_percentage"],
                similarity_score=original_job.get("similarity"),
                keywords=original_job.get("keywords"),
                description=original_job.get("description")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app4:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )