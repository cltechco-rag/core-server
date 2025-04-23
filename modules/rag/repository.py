from sqlalchemy import func, text, cast
from sqlalchemy.orm import aliased
from pgvector.sqlalchemy import Vector
from core.database import SessionLocal
from models.rag_document import RAGDocument
from models.video import Video
import logging

logger = logging.getLogger(__name__)


class RAGRepository:
    def __init__(self) -> None:
        self.db_factory = SessionLocal

    def create_document(
        self,
        video_id: int,
        title: str,
        content: str,
        tokenized_text: str,
        content_embedding: any,
        title_embedding: any,
    ) -> RAGDocument:
        db = self.db_factory()
        try:
            document = RAGDocument(
                video_id=video_id,
                content=content,
                title=title,
                tokenized_text=tokenized_text,
                content_embedding=content_embedding,
                title_embedding=title_embedding,
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            return document
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating document: {str(e)}", exc_info=True)
            raise
        finally:
            db.close()

    def get_scores(self, user_id, query, query_embedding):
        db = self.db_factory()
        try:
            # First get a count of documents to limit the query if needed
            VideoAlias = aliased(Video)
            raw_results = (
                db.query(
                    RAGDocument,
                    (
                        1
                        - func.cosine_distance(
                            RAGDocument.content_embedding,
                            cast(query_embedding, Vector),
                        )
                    ).label("content_similarity"),
                    (
                        1
                        - func.cosine_distance(
                            RAGDocument.title_embedding,
                            cast(query_embedding, Vector),
                        )
                    ).label("title_similarity"),
                    func.ts_rank_cd(
                        text("to_tsvector('simple', tokenized_text)"),
                        text("plainto_tsquery('simple', :query)"),
                        32,  # normalization option
                    ).label("bm25_score"),
                )
                .join(VideoAlias, RAGDocument.video_id == VideoAlias.id)
                .filter(VideoAlias.user_id == user_id)
                .params(query=query, query_embedding=query_embedding)
                .all()
            )

            return raw_results
        except Exception as e:
            logger.error(f"Error in get_scores: {str(e)}", exc_info=True)
            raise
        finally:
            db.close()

    def delete_documents_by_video_id(self, video_id: int) -> int:
        """
        Delete all RAG documents associated with a specific video ID.

        Args:
            video_id (int): The ID of the video whose documents should be deleted

        Returns:
            int: The number of documents deleted
        """
        db = self.db_factory()
        try:
            result = (
                db.query(RAGDocument).filter(RAGDocument.video_id == video_id).delete()
            )
            db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(
                f"Error deleting documents for video {video_id}: {str(e)}",
                exc_info=True,
            )
            raise
        finally:
            db.close()
