"""Part D -- the same policy RAG pipeline built from LangChain components (langchain-core).

Run from the repository root:  python part_d/rag_langchain.py
Still fully offline: the "LLM" is mock_llm.py wrapped as a LangChain chat model, and the
retriever is the TF-IDF index from part_d/rag.py wrapped as a LangChain retriever. The script
proves the chain gives exactly the same answers and sources as the plain pipeline.

    question --> TfidfPolicyRetriever --> build CONTEXT/QUESTION message --> ChatPromptTemplate
             --> MockChatModel (mock_llm.chat) --> StrOutputParser --> answer
"""
import os
import sys
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm                                   # noqa: E402
import rag                                        # noqa: E402

ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


class MockChatModel(BaseChatModel):
    """LangChain chat model whose 'inference' is the offline mock_llm.chat()."""

    @property
    def _llm_type(self) -> str:
        return "instamart-mock-llm"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        payload = [{"role": ROLE[m.type], "content": m.content} for m in messages]
        resp = mock_llm.chat(payload)
        msg = resp.choices[0].message
        usage = resp.usage
        ai = AIMessage(content=msg.content or "",
                       usage_metadata={"input_tokens": usage.prompt_tokens,
                                       "output_tokens": usage.completion_tokens,
                                       "total_tokens": usage.total_tokens})
        return ChatResult(generations=[ChatGeneration(message=ai)])


class TfidfPolicyRetriever(BaseRetriever):
    """LangChain retriever over the TF-IDF PolicyIndex from rag.py."""
    index: Any
    k: int = 3

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return [Document(page_content=c["text"], metadata={"source": c["source"], "score": c["score"]})
                for c in self.index.retrieve(query, k=self.k)]


PROMPT = ChatPromptTemplate.from_messages([("system", rag.SYSTEM_PROMPT), ("human", "{user_message}")])


def build_chain(index=None):
    retriever = TfidfPolicyRetriever(index=index or rag.default_index())
    to_message = RunnableLambda(lambda x: {"user_message": rag.build_user_message(
        [{"source": d.metadata["source"], "text": d.page_content} for d in x["docs"]], x["question"])})
    answer_chain = to_message | PROMPT | MockChatModel() | StrOutputParser()
    # run retrieval once, then produce the answer and keep the sources alongside it
    return (RunnableParallel(docs=retriever, question=RunnablePassthrough())
            | RunnableParallel(answer=answer_chain,
                               sources=RunnableLambda(lambda x: [d.metadata["source"] for d in x["docs"]])))


def answer_langchain(question, index=None):
    return build_chain(index).invoke(question)


def main():
    print("## LangChain RAG chain vs plain pipeline\n")
    for label, index in [("overlap 10", rag.default_index()), ("overlap 0", rag.PolicyIndex(overlap=0))]:
        ev_plain = rag.evaluate(index)
        ev_chain = rag.evaluate(index, answer_fn=answer_langchain)
        same = (ev_plain["answer"] == ev_chain["answer"]).all()
        s = rag.summarise(ev_chain)
        print(f"{label}: LangChain chain -> doc in top 3 {s['doc_in_top3']}/{s['answerable']}, "
              f"fact in answer {s['fact_in_answer']}/{s['answerable']}, fallback {s['fallback']}/"
              f"{s['unanswerable']}; identical to plain pipeline on all {len(ev_plain)} questions: {same}")
        assert same
    out = answer_langchain("Is cash on delivery available for a large order?")
    print(f"\nexample: {out['answer']}  sources={out['sources']}")
    print("\nLANGCHAIN CHAIN MATCHES PLAIN PIPELINE")


if __name__ == "__main__":
    main()
