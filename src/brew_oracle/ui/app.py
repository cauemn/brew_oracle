import streamlit as st

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator


@st.cache_resource
def get_orchestrator() -> BrewingOrchestrator:
    return BrewingOrchestrator(hybrid=True, rerank=True)


def main() -> None:
    st.set_page_config(page_title="Brew Oracle", page_icon="🍺", layout="wide")
    st.title("Brew Oracle")

    orchestrator = get_orchestrator()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Pergunta", placeholder="Como posso ajudar?")
        submitted = st.form_submit_button("Consultar", type="primary")
        if submitted and question.strip():
            with st.spinner("Consultando..."):
                text, refs = orchestrator.ask_with_refs(question)
            st.session_state.messages.append(
                {"question": question, "answer": text, "refs": refs}
            )

    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            if msg["refs"]:
                st.markdown("**Referências**")
                for ref in msg["refs"]:
                    st.markdown(f"- {ref}")


if __name__ == "__main__":
    main()
