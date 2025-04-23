import logging
from config.config import settings
from ..upload.repository import UploadRepository
from .repository import RAGRepository
from .schema import AnswerResponse
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from konlpy.tag import Okt
import time
from typing import Dict, Any
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)

# 백그라운드 작업 상태 추적을 위한 전역 딕셔너리
background_tasks_status: Dict[int, Dict[str, Any]] = {}


TOPIC_SPLIT_PROMPT = """
You are an AI that segments lecture transcripts by topic.

Split the transcript into a list of text chunks, each corresponding to a distinct topic or shift in discussion. Do not summarize or label the topics. An ideal length of each chunk is 2000 characters, but it can be slightly shorter or longer if necessary.

Only return a list where each item is a string containing one chunk of transcript. No bullet points or numbering—just the list format.

Transcript:
{transcript}
"""


class RAGService:

    def __init__(self, db: Session) -> None:

        self.llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=settings.AZURE_CHAT_DEPLOYMENT,
            openai_api_version=settings.AZURE_CHAT_API_VERSION,
            openai_api_key=settings.AZURE_CHAT_API_KEY,
            azure_endpoint=settings.AZURE_CHAT_ENDPOINT,
        )
        self.okt = Okt()
        self.client = AzureOpenAI(
            api_key=settings.AZURE_CHAT_API_KEY,
            api_version=settings.AZURE_CHAT_API_VERSION,
            azure_endpoint=settings.AZURE_CHAT_ENDPOINT,
        )
        self.upload_repo = UploadRepository(db)
        self.rag_repo = RAGRepository()
        self.db = db

    def _split_by_topic_llm(self, transcript: str, chunk_len: int = 6000) -> list[str]:
        prompt = PromptTemplate.from_template(TOPIC_SPLIT_PROMPT)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        chunks_by_topic = []
        # Split the transcript into chunks by chunk_len
        for i in range(0, len(transcript), chunk_len):
            chunk = transcript[i : i + chunk_len]
            response = chain.run({"transcript": chunk})
            # Parse the response into a list of chunks
            try:
                chunks_by_topic.extend(eval(response))
            except:
                pass
        return chunks_by_topic

    def chunk_transcript(self, transcript: str, chunk_size=1000):
        """
        Splits the transcript into smaller chunks for processing.
        """
        # Use the LLM to split by topic
        chunks_by_topic = self._split_by_topic_llm(transcript, chunk_size)
        # Further split each topic chunk into smaller chunks
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
        )
        for topic_chunk in chunks_by_topic:
            # Split each topic chunk into smaller chunks
            all_chunks.extend(text_splitter.split_text(topic_chunk))
        return all_chunks

    def preprocess_korean_text(self, text):
        """한국어 텍스트 전처리 및 토큰화 함수"""
        # Okt로 형태소 분석 및 토큰화
        tokens = []
        for word, pos in self.okt.pos(text):
            # 명사, 동사, 형용사, 부사만 추출
            if pos in ["Noun", "Verb", "Adjective", "Adverb"]:
                tokens.append(word)

        # 한 글자 토큰 제거
        return " ".join(tokens)

    def run_vector_embedding(self, text: str):
        content_embedding = (
            self.client.embeddings.create(
                model=settings.AZURE_EMBEDDING_DEPLOYMENT, input=text
            )
            .data[0]
            .embedding
        )
        return content_embedding

    def get_background_task_status(self, video_id: int) -> Dict[str, Any]:
        """백그라운드 작업 상태를 반환합니다."""
        if video_id not in background_tasks_status:
            return {
                "video_id": video_id,
                "status": "unknown",
                "message": "작업 상태를 찾을 수 없습니다.",
            }
        return {"video_id": video_id, **background_tasks_status[video_id]}

    def run_preprocess_pipeline(self, video_id: int):
        """텍스트를 벡터 임베딩으로 변환하고 토큰화하는 백그라운드 작업"""
        try:
            self.upload_repo.update_video(video_id=video_id, rag_status="PENDING")
            self.db.commit()

            # 작업 상태 초기화
            background_tasks_status[video_id] = {
                "status": "processing",
                "start_time": time.time(),
                "progress": 0,
                "last_update": time.time(),
                "log_messages": ["작업이 시작되었습니다."],
            }

            video = self.upload_repo.get_video_by_id(video_id)
            if not video:
                error_msg = f"비디오 ID {video_id}를 찾을 수 없습니다."
                logger.error(error_msg)
                background_tasks_status[video_id]["status"] = "failed"
                background_tasks_status[video_id]["log_messages"].append(error_msg)
                return False

            if not video.transcript:
                error_msg = f"비디오 ID {video_id}에 대한 기록이 없습니다."
                logger.error(error_msg)
                background_tasks_status[video_id]["status"] = "failed"
                background_tasks_status[video_id]["log_messages"].append(error_msg)
                return False

            logger.info(f"비디오 ID {video_id}의 텍스트를 벡터 임베딩으로 변환 중...")
            background_tasks_status[video_id]["log_messages"].append(
                f"비디오 ID {video_id}의 텍스트를 벡터 임베딩으로 변환 중..."
            )

            content = video.transcript.content
            background_tasks_status[video_id]["progress"] = 20

            # 한국어 텍스트 전처리 및 토큰화
            chunks = self.chunk_transcript(content)
            background_tasks_status[video_id]["progress"] = 40
            background_tasks_status[video_id]["log_messages"].append(
                f"텍스트를 {len(chunks)}개의 청크로 분할했습니다."
            )

            for i, chunk in enumerate(chunks):
                try:
                    tokenized_text = self.preprocess_korean_text(chunk)
                    content_embedding = self.run_vector_embedding(chunk)
                    title_embedding = self.run_vector_embedding(video.title)
                    self.rag_repo.create_document(
                        video_id=video.id,
                        title=video.title,
                        content=chunk,
                        tokenized_text=tokenized_text,
                        content_embedding=content_embedding,
                        title_embedding=title_embedding,
                    )
                    progress = 40 + (i / len(chunks)) * 60
                    background_tasks_status[video_id]["progress"] = progress
                    background_tasks_status[video_id]["last_update"] = time.time()
                except Exception as e:
                    error_msg = f"청크 {i} 처리 중 오류 발생: {str(e)}"
                    logger.error(error_msg)
                    background_tasks_status[video_id]["log_messages"].append(error_msg)
                    raise

            background_tasks_status[video_id]["status"] = "completed"
            background_tasks_status[video_id]["progress"] = 100
            background_tasks_status[video_id]["log_messages"].append(
                "작업이 완료되었습니다."
            )
            self.upload_repo.update_video(video_id=video_id, rag_status="INCLUDED")
            self.db.commit()
            return True

        except Exception as e:
            error_msg = f"전처리 파이프라인 실행 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            background_tasks_status[video_id]["status"] = "failed"
            background_tasks_status[video_id]["log_messages"].append(error_msg)
            self.upload_repo.update_video(video_id=video_id, rag_status="NOT_INCLUDED")
            self.db.commit()
            return False

    def run_hybrid_search(
        self,
        query: str,
        content_vector_weight: float,
        title_vector_weight: float,
        bm25_weight: float,
        user_id: int,
        top_k: int = 10,
    ):
        start_time = time.time()
        query_embedding = self.run_vector_embedding(query)
        raw_results = self.rag_repo.get_scores(user_id, query, query_embedding)

        if not raw_results:
            return [], 0

        # Get min and max values for normalization
        content_similarities = [r[1] for r in raw_results]
        title_similarities = [r[2] for r in raw_results]
        bm25_scores = [r[3] for r in raw_results]

        content_min = min(content_similarities)
        content_max = max(content_similarities)
        title_min = min(title_similarities)
        title_max = max(title_similarities)
        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)

        # Calculate normalized and weighted scores
        results = []
        for doc, content_sim, title_sim, bm25_score in raw_results:
            # Normalize scores
            norm_content = (
                (content_sim - content_min) / (content_max - content_min)
                if content_max != content_min
                else 0
            )
            norm_title = (
                (title_sim - title_min) / (title_max - title_min)
                if title_max != title_min
                else 0
            )
            norm_bm25 = (
                (bm25_score - bm25_min) / (bm25_max - bm25_min)
                if bm25_max != bm25_min
                else 0
            )

            # Apply weights
            weighted_score = (
                content_vector_weight * norm_content
                + title_vector_weight * norm_title
                + bm25_weight * norm_bm25
            )
            results.append((doc, weighted_score))

        # Sort by weighted score and get top_k
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:top_k]

        logger.info(f"Found {len(final_results)} results")
        query_time = time.time() - start_time
        return final_results, query_time

    # def search(self, query: str, top_k: int = 10) -> SearchResponse:
    #     """Perform hybrid search and return formatted results"""
    #     results, query_time = self.repository.hybrid_search(query, top_k)

    #     search_results = [
    #         SearchResult(
    #             id=doc.id,
    #             title=doc.title,
    #             content=doc.content,
    #             score=score,
    #             created_at=getattr(doc, "created_at", datetime.now()),
    #         )
    #         for doc, score in results
    #     ]

    #     return SearchResponse(
    #         results=search_results,
    #         total_results=len(search_results),
    #         query_time=query_time,
    #     )

    async def generate_answer(
        self,
        query: str,
        content_vector_weight: float,
        title_vector_weight: float,
        bm25_weight: float,
        rerank: bool,
        user_id: int,
        top_k: int = 5,
    ) -> AnswerResponse:
        """Generate an answer using LLM based on retrieved documents"""
        # Retrieve relevant documents
        results, query_time = self.run_hybrid_search(
            query,
            content_vector_weight,
            title_vector_weight,
            bm25_weight,
            user_id,
            top_k,
        )

        print(results)

        # Prepare context from retrieved documents
        context = "\n\n".join(
            [f"문서 제목: {doc.title}\n내용: {doc.content}" for doc, _ in results]
        )

        prompt = ChatPromptTemplate.from_template(
            """다음은 검색된 문서 내용입니다:

            {context}

            다음 질문에 대해 문서 내용을 바탕으로 답변해주세요:
            {question}

            답변은 한국어로 작성해주세요. 문서 내용을 바탕으로 구체적이고 짧고 명확하게 답변해주세요.
            """
        )
        chain = prompt | self.llm | StrOutputParser()

        # Generate answer using LLM
        answer = await chain.ainvoke({"context": context, "question": query})

        return AnswerResponse(
            answer=answer, query_time=query_time, total_documents=len(results)
        )

    def remove_video_from_rag_search(self, video_id):

        self.rag_repo.delete_documents_by_video_id(video_id)
        self.upload_repo.update_video(video_id=video_id, rag_status="NOT_INCLUDED")
        self.db.commit()
