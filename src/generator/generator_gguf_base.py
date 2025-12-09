from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import time
import os

from src.utils.config import RAGConfig
from src.router.query_router import QueryRouter
from src.prompts.dynamic_prompts import PromptManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFGenerator:
    """
    GGUF 기반 Llama-3 생성기
    
    llama.cpp를 사용하여 GGUF 포맷 모델을 로드하고
    입찰 관련 질의응답을 수행합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 8192,
        n_threads: int = 8,
        config = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "당신은 RFP(제안요청서) 분석 및 요약 전문가입니다."
    ):
        """생성기 초기화"""
        self.config = config or RAGConfig() 
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 모델 (나중에 로드)
        self.model = None
        
        logger.info(f"GGUFGenerator 초기화 완료 (Base 모델)")
    
    def load_model(self) -> None:
        """
        GGUF 모델 로드
        
        ✅ Base 모델 사용: Config에서 BASE_MODEL_HUB_REPO 가져오기
        """
        
        # 중복 로드 방지
        if self.model is not None:
            logger.info("모델이 이미 로드되어 있습니다.")
            return
        
        try:
            # Config에서 USE_MODEL_HUB 확인
            use_model_hub = getattr(self.config, 'USE_MODEL_HUB', True)
            
            # Model Hub 사용 여부에 따라 경로 결정
            if use_model_hub:
                # === Model Hub에서 다운로드 ===
                # ✅ Config에서 Base 모델 정보 가져오기
                base_model_repo = getattr(
                    self.config, 
                    'BASE_MODEL_HUB_REPO', 
                    'Dongjin1203/Llama-3-Open-Ko-8B-GGUF'
                )
                base_model_filename = getattr(
                    self.config, 
                    'BASE_MODEL_HUB_FILENAME', 
                    'Llama-3-Open-Ko-8B-Q4_K_M.gguf'
                )
                model_cache_dir = getattr(self.config, 'MODEL_CACHE_DIR', '.cache/models')
                
                logger.info(f"📥 Base 모델 다운로드: {base_model_repo}")
                logger.info(f"   파일명: {base_model_filename}")
                
                from huggingface_hub import hf_hub_download
                
                model_path = hf_hub_download(
                    repo_id=base_model_repo,
                    filename=base_model_filename,
                    cache_dir=model_cache_dir,
                    local_dir=model_cache_dir,
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"✅ Base 모델 다운로드 완료: {model_path}")
                
            else:
                # === 로컬 파일 사용 ===
                model_path = self.model_path
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"❌ 로컬 모델 파일을 찾을 수 없습니다: {model_path}\n"
                        f"   USE_MODEL_HUB=true로 설정하거나 모델 파일을 준비하세요."
                    )
                
                logger.info(f"📂 로컬 Base 모델 사용: {model_path}")
            
            # === 공통: 모델 로드 ===
            logger.info(f"🚀 Base GGUF 모델 로드 중...")
            logger.info(f"   GPU 레이어: {self.n_gpu_layers}")
            logger.info(f"   컨텍스트: {self.n_ctx}")
            
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=True,
            )
            
            # 실제 적용된 n_ctx 확인
            actual_n_ctx = self.model.n_ctx()
            logger.info("✅ Base GGUF 모델 로드 완료!")
            logger.info(f"   - 모델: {base_model_repo if use_model_hub else 'local'}")
            logger.info(f"   - 설정한 n_ctx: {self.n_ctx}")
            logger.info(f"   - 실제 n_ctx: {actual_n_ctx}")
            
            if actual_n_ctx < self.n_ctx:
                logger.warning(f"⚠️ n_ctx가 예상보다 작습니다: {actual_n_ctx} < {self.n_ctx}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise RuntimeError(f"모델 로드 중 오류 발생: {e}")
    
    def format_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """GGUF 모델용 간단한 프롬프트 포맷팅"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if context is not None:
            user_message = f"참고 문서:\n{context}\n\n질문: {question}"
        else:
            user_message = question
        
        formatted_prompt = f"""### 시스템
{system_prompt}

### 사용자
{user_message}

### 답변
"""
        
        return formatted_prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """프롬프트를 입력받아 응답 생성"""
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        try:
            logger.info(f"🔄 생성 시작 (max_tokens={max_new_tokens}, temp={temperature})")
            start_time = time.time()
            
            output = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=[
                    "###", "\n\n###", 
                    "### 사용자", "\n사용자:", 
                    "</s>",
                    "한국어 답변", "한국어로 답변", "지침:",
                    "문장", "(문장",
                    "\n\n",
                    "?",
                    "요?", "까?", "나요?", "습니까?"
                ],
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 생성 완료: {elapsed:.2f}초")
            
            response = output['choices'][0]['text'].strip()
            
            logger.info(f"📝 응답 길이: {len(response)} 글자")
            return response
            
        except Exception as e:
            logger.error(f"❌ 생성 중 오류 발생: {e}")
            raise RuntimeError(f"텍스트 생성 실패: {e}")
    
    def chat(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt=None,
        **kwargs
    ) -> str:
        """질문에 대한 응답 생성"""
        prompt = self.format_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt
        )
        
        response = self.generate(prompt, **kwargs)
        
        return response


class GGUFBaseRAGPipeline:
    """
    Base 모델 + RAG 파이프라인
    
    ✅ Base 모델 사용 (beomi/Llama-3-Open-Ko-8B)
    ✅ RAG 유지
    ✅ 기존 generator_gguf.py와 동일한 기능
    """
    
    def __init__(
        self,
        config=None,
        model: str = None,
        top_k: int = None,
        n_gpu_layers: int = None,
        n_ctx: int = None,
        n_threads: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        search_mode: str = None,
        alpha: float = None
    ):
        """초기화"""
        self.config = config or RAGConfig()
        
        # 검색 설정
        self.top_k = top_k or getattr(self.config, 'DEFAULT_TOP_K', 10)
        self.search_mode = search_mode or getattr(self.config, 'DEFAULT_SEARCH_MODE', 'hybrid_rerank')
        self.alpha = alpha if alpha is not None else getattr(self.config, 'DEFAULT_ALPHA', 0.5)
        
        # Retriever 초기화
        logger.info("RAGRetriever 초기화 중...")
        from src.retriever.retriever import RAGRetriever
        self.retriever = RAGRetriever(config=self.config)
        
        # GGUF 설정
        gguf_n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else getattr(self.config, 'GGUF_N_GPU_LAYERS', 35)
        gguf_n_ctx = n_ctx if n_ctx is not None else getattr(self.config, 'GGUF_N_CTX', 2048)
        gguf_n_threads = n_threads if n_threads is not None else getattr(self.config, 'GGUF_N_THREADS', 4)
        gguf_max_new_tokens = max_new_tokens if max_new_tokens is not None else getattr(self.config, 'GGUF_MAX_NEW_TOKENS', 512)
        gguf_temperature = temperature if temperature is not None else getattr(self.config, 'GGUF_TEMPERATURE', 0.7)
        gguf_top_p = top_p if top_p is not None else getattr(self.config, 'GGUF_TOP_P', 0.9)
        
        # 모델 경로 (사용 안 함, Hub에서 다운로드)
        gguf_model_path = getattr(self.config, 'GGUF_MODEL_PATH', '.cache/models/llama-3-ko-8b.gguf')
        
        # 시스템 프롬프트
        system_prompt = getattr(self.config, 'SYSTEM_PROMPT', '당신은 한국 공공기관 사업제안서 분석 전문가입니다.')
        
        # GGUFGenerator 초기화
        logger.info("GGUFGenerator 초기화 중... (Base 모델)")
        logger.info(f"   GPU 레이어: {gguf_n_gpu_layers}")
        logger.info(f"   컨텍스트: {gguf_n_ctx}")
        
        self.generator = GGUFGenerator(
            model_path=gguf_model_path,
            n_gpu_layers=gguf_n_gpu_layers,
            n_ctx=gguf_n_ctx,
            n_threads=gguf_n_threads,
            config=self.config,
            max_new_tokens=gguf_max_new_tokens,
            temperature=gguf_temperature,
            top_p=gguf_top_p,
            system_prompt=system_prompt
        )
        
        # 모델 로드
        logger.info("Base GGUF 모델 로드 중...")
        self.generator.load_model()
        
        # Router
        self.router = QueryRouter()
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        # 마지막 검색 결과
        self._last_retrieved_docs = []
        
        logger.info("✅ GGUFBaseRAGPipeline 초기화 완료")
        logger.info(f"   - 검색 모드: {self.search_mode}")
        logger.info(f"   - 기본 top_k: {self.top_k}")
    
    def _retrieve_and_format(self, query: str) -> str:
        """검색 수행 및 컨텍스트 포맷팅"""
        # 검색 모드에 따라 문서 검색
        if self.search_mode == "embedding":
            docs = self.retriever.search(query, top_k=self.top_k)
        elif self.search_mode == "embedding_rerank":
            docs = self.retriever.search_with_rerank(query, top_k=self.top_k)
        elif self.search_mode == "hybrid":
            docs = self.retriever.hybrid_search(
                query, top_k=self.top_k, alpha=self.alpha
            )
        elif self.search_mode == "hybrid_rerank":
            docs = self.retriever.hybrid_search_with_rerank(
                query, top_k=self.top_k, alpha=self.alpha
            )
        else:
            docs = self.retriever.search(query, top_k=self.top_k)
        
        # 마지막 검색 결과 저장
        self._last_retrieved_docs = docs
        
        # 컨텍스트 포맷팅
        return self._format_context(docs)
    
    def _format_context(self, retrieved_docs: list) -> str:
        """검색된 문서를 컨텍스트로 변환"""
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        max_context_chars = 8000
        
        current_length = 0
        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = f"[문서 {i}]\n{doc['content']}\n"
            doc_length = len(doc_text)
            
            if current_length + doc_length > max_context_chars:
                logger.warning(f"⚠️ 컨텍스트 길이 제한: {i-1}개 문서만 사용")
                break
            
            context_parts.append(doc_text)
            current_length += doc_length
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: list) -> list:
        """검색된 문서를 sources 형식으로 변환"""
        sources = []
        for doc in retrieved_docs:
            source_info = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'filename': doc.get('filename', 'N/A'),
                'organization': doc.get('organization', 'N/A')
            }
            
            if 'rerank_score' in doc:
                source_info['score'] = doc['rerank_score']
                source_info['score_type'] = 'rerank'
            elif 'hybrid_score' in doc:
                source_info['score'] = doc['hybrid_score']
                source_info['score_type'] = 'hybrid'
            elif 'relevance_score' in doc:
                source_info['score'] = doc['relevance_score']
                source_info['score_type'] = 'embedding'
            else:
                source_info['score'] = 0
                source_info['score_type'] = 'unknown'
            
            sources.append(source_info)
        
        return sources
    
    def _estimate_usage(self, query: str, answer: str) -> dict:
        """토큰 사용량 추정"""
        prompt_tokens = len(query.split()) * 2
        completion_tokens = len(answer.split()) * 2
        
        return {
            'total_tokens': prompt_tokens + completion_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    
    def generate_answer(
        self,
        query: str,
        top_k: int = None,
        search_mode: str = None,
        alpha: float = None
    ) -> dict:
        """답변 생성 (Base 모델 + RAG)"""
        try:
            start_time = time.time()
            
            # 파라미터 설정
            if top_k is not None:
                self.top_k = top_k
            if search_mode is not None:
                self.search_mode = search_mode
            if alpha is not None:
                self.alpha = alpha

            # Router로 검색 여부 결정
            classification = self.router.classify(query)
            query_type = classification['type']
            
            logger.info(f"📍 분류: {query_type} (신뢰도: {classification['confidence']:.2f})")
            
            # 타입별 처리
            if query_type in ['greeting', 'thanks', 'out_of_scope']:
                # 검색 스킵
                context = None
                used_retrieval = False
                self._last_retrieved_docs = []
                
                # 동적 프롬프트
                system_prompt = PromptManager.get_prompt(query_type, model_type="gguf")
                logger.info(f"⏭️ RAG 스킵: {query_type}")
            
            elif query_type == 'document':
                # RAG 수행
                context = self._retrieve_and_format(query)
                used_retrieval = True
                
                # 동적 프롬프트
                system_prompt = PromptManager.get_prompt('document', model_type="gguf")
                logger.info(f"🔍 RAG 수행: {len(self._last_retrieved_docs)}개 문서")
            
            # 답변 생성
            answer = self.generator.chat(
                question=query,
                context=context,
                system_prompt=system_prompt
            )
            
            elapsed_time = time.time() - start_time
            
            # 대화 히스토리 추가
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # 결과 반환
            return {
                'answer': answer,
                'sources': self._format_sources(self._last_retrieved_docs),
                'used_retrieval': used_retrieval,
                'query_type': query_type,
                'search_mode': self.search_mode if used_retrieval else 'direct',
                'routing_info': classification,
                'elapsed_time': elapsed_time,
                'usage': self._estimate_usage(query, answer)
            }
        
        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e
    
    def chat(self, query: str) -> str:
        """간단한 대화 인터페이스"""
        result = self.generate_answer(query)
        return result['answer']
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        logger.info("🗑️ 대화 히스토리가 초기화되었습니다.")
    
    def get_history(self) -> List[Dict]:
        """대화 히스토리 반환"""
        return self.chat_history.copy()
    
    def set_search_config(
        self,
        search_mode: str = None,
        top_k: int = None,
        alpha: float = None
    ):
        """검색 설정 변경"""
        if search_mode is not None:
            self.search_mode = search_mode
        if top_k is not None:
            self.top_k = top_k
        if alpha is not None:
            self.alpha = alpha
        
        logger.info(
            f"🔧 검색 설정 변경: mode={self.search_mode}, "
            f"top_k={self.top_k}, alpha={self.alpha}"
        )