import os

from dotenv import load_dotenv

from get_answer import get_answer

load_dotenv()

import streamlit as st

from build_knowledgebase import build_knowledgebase


if __name__ == '__main__':
    # ----------------------
    # Streamlit UI
    # ----------------------
    st.set_page_config(page_title="RKO - Repo Knockout", layout="centered")

    st.title("🥊 RKO - Repo Knockout")
    st.markdown("#### Understand Github Repository with AI")

    # Session state initialization
    if "stage" not in st.session_state:
        st.session_state.stage = "input"
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "query_asked" not in st.session_state:
        st.session_state.query_asked = False
    if "repo_exists" not in st.session_state:
        st.session_state.repo_exists = False

    # Stage: Input
    if st.session_state.stage == "input":
        repo_url = st.text_input("Enter GitHub Repo URL")
        branch_name = st.text_input("Enter branch")

        if st.button("Run") and repo_url and branch_name:
            st.session_state.repo_url = repo_url
            st.session_state.branch_name = branch_name
            st.session_state.stage = "processing"
            st.rerun()

    # Stage: Processing
    elif st.session_state.stage == "processing":
        if not st.session_state.processed:
            st.info("🔄 Fetching repo and building knowledge base...")
            result = build_knowledgebase(st.session_state.repo_url, st.session_state.branch_name)

            if result == "already_exists":
                st.session_state.repo_exists = True
            else:
                st.session_state.repo_exists = False

            st.session_state.processed = True
            st.session_state.stage = "qa"
            st.rerun()
        else:
            st.info("✅ Processing complete. Moving to Q&A...")

    # Stage: Q&A
    elif st.session_state.stage == "qa":
        if st.session_state.repo_exists:
            st.warning("⚠️ This repository is already in the knowledge base. Showing existing results.")
            st.session_state.repo_exists = False  # Reset so it shows only once

        st.markdown("### Ask anything about the repo:")

        if not st.session_state.query_asked:
            query = st.text_input("Your question", key="qa_input")
            if st.button("Ask") and query:
                with st.spinner("Thinking..."):
                    answer = get_answer(query)
                    st.session_state.last_answer = answer
                    st.session_state.query_asked = True
                    st.rerun()
        else:
            st.markdown("#### 💡 Answer:")
            st.write(st.session_state.last_answer)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔁 Ask Another Question"):
                    st.session_state.query_asked = False
                    st.rerun()
            with col2:
                if st.button("🔄 New Repo"):
                    # Reset all session states
                    st.session_state.stage = "input"
                    st.session_state.processed = False
                    st.session_state.query_asked = False
                    st.session_state.repo_exists = False
                    st.rerun()