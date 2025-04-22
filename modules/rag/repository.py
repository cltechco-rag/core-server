from sqlalchemy import func, text, cast, bindparam
from pgvector.sqlalchemy import Vector
from core.database import SessionLocal
from models.rag_document import RAGDocument
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

    def get_scores(self, query, query_embedding):
        db = self.db_factory()
        try:
            # First get a count of documents to limit the query if needed
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
                .params(query=query, query_embedding=query_embedding)
                .all()
            )

            return raw_results
        except Exception as e:
            logger.error(f"Error in get_scores: {str(e)}", exc_info=True)
            raise
        finally:
            db.close()
