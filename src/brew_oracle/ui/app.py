import streamlit as st

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator


@st.cache_resource
def get_orchestrator() -> BrewingOrchestrator:
    return BrewingOrchestrator()


def main() -> None:
    st.title("Brew Oracle")
    orchestrator = get_orchestrator()

    if "response" not in st.session_state:
        st.session_state.response = ""
        st.session_state.refs = []

    question = st.text_area("Pergunta")
    if st.button("Consultar") and question.strip():
        text, refs = orchestrator.ask_with_refs(question)
        st.session_state.response = text
        st.session_state.refs = refs

    if st.session_state.response:
        st.markdown(st.session_state.response)
        if st.session_state.refs:
            st.markdown("## Referências")
            for ref in st.session_state.refs:
                st.markdown(f"- {ref}")


if __name__ == "__main__":
    main()
