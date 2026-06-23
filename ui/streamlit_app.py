"""
Streamlit Web Interface

Provides a slick dashboard interface to interact with your compiled LangGraph workflow.
Supports multi-turn memory threads, raw SQL debugging view layouts, and Plotly chart metrics.
"""

import logging
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import compiled graph app, utilities, and structured response definitions from the app package
from app.graph import app
from app.utils.dataframe_utils import query_results_to_dataframe
from app.models.response import Text2SQLResponse, ChatResponse, VisualizationResponse

# Set up browser page styling configurations
st.set_page_config(
    page_title="Omnichannel Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_agent_payload(response) -> None:
    """
    Renders structured multi-agent outputs cleanly onto the Streamlit view layer.
    """
    if not response:
        st.warning("Empty response received from the agent workspace pipeline.")
        return

    # -------------------------------------------------------------------------
    # Scenario A: Handle Failures or Explicit Context Clarifications
    # -------------------------------------------------------------------------
    if hasattr(response, "success") and not response.success:
        st.error(f"**Operational Alert:** {response.message}")
        if getattr(response, "error", None):
            st.caption(f"Error Diagnostic Signature: {response.error}")
        return

    # -------------------------------------------------------------------------
    # Scenario B: Text2SQL Node Render Engine Output
    # -------------------------------------------------------------------------
    if isinstance(response, Text2SQLResponse):
        st.success(response.message)
        
        with st.expander("🛠️ View Compiled PostgreSQL Syntax", expanded=False):
            st.code(response.sql_query, language="sql")

        if response.query_results:
            df = query_results_to_dataframe(response.query_results)
            if df is not None and not df.empty:
                st.subheader("Data Metrics View")
                st.dataframe(df, use_container_width=True)
                
                # Provide a clean pipeline download link for deeper analytical processing
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Dataset as CSV",
                    data=csv_bytes,
                    file_name="marketing_metrics_extract.csv",
                    mime="text/csv"
                )

    # -------------------------------------------------------------------------
    # Scenario C: Visualization Node Render Engine Output
    # -------------------------------------------------------------------------
    elif isinstance(response, VisualizationResponse):
        st.success(response.message)
        
        df = query_results_to_dataframe({"success": True, "rows": response.data_payload})
        if df is not None and not df.empty:
            st.subheader(f"📈 {response.title}")
            
            try:
                chart_type_lower = response.chart_type.lower()
                
                if "bar" in chart_type_lower:
                    st.bar_chart(data=df, x=response.x_axis, y=response.y_axes, use_container_width=True)
                elif "line" in chart_type_lower:
                    st.line_chart(data=df, x=response.x_axis, y=response.y_axes, use_container_width=True)
                elif "area" in chart_type_lower:
                    st.area_chart(data=df, x=response.x_axis, y=response.y_axes, use_container_width=True)
                else:
                    # Fallback to interactive data view layout if chart form type is exotic
                    st.dataframe(df, use_container_width=True)
                    
            except Exception as chart_err:
                st.error(f"Failed to generate layout parameters: {str(chart_err)}")
        
        if getattr(response, "explanation", None):
            st.info(f"**Data Scientist Assessment:** {response.explanation}")

    # -------------------------------------------------------------------------
    # Scenario D: Conversational General Chat Node Output
    # -------------------------------------------------------------------------
    elif isinstance(response, ChatResponse):
        st.markdown(response.answer)


def main():
    """Builds layout scaffolds and hooks context triggers down to the Graph engine."""
    
    # Initialize long-lived application state stores
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())

    # Sidebar Structural Design
    with st.sidebar:
        st.title("🤖 Engine Controls")
        st.caption(f"Active Session Thread ID: `{st.session_state.thread_id}`")
        st.markdown("---")
        st.markdown(
            """
            ### Analytical Capabilities
            This unified system evaluates omnichannel marketing metrics pipelines:
            * **Conversational Chat:** Inquire about schema constraints or high-level goals.
            * **Automated Text2SQL:** Extracts live performance numbers dynamically out of your PostgreSQL data pools.
            * **Visual Mapping Layouts:** Generates declarative chart specifications automatically.
            """
        )
        st.markdown("---")
        if st.button("🔄 Clear Active Thread History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Omnichannel Marketing Analytics Hub")
    st.markdown("Query integrated multi-channel performance metrics using intuitive natural language interfaces.")

    # Render history message frames present in state
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            if role == "user":
                st.markdown(msg.content)
            else:
                payload = getattr(msg, "response_payload", None) or msg.content
                if isinstance(payload, (Text2SQLResponse, ChatResponse, VisualizationResponse)):
                    display_agent_payload(payload)
                else:
                    st.markdown(str(msg.content))

    # Accept incoming interaction queries from text field hooks
    if prompt := st.chat_input("Ask a question (e.g., 'Compare total impressions vs spending by channel last quarter')"):
        
        # 1. Commit and print the incoming human request message frame
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Invoke our unified LangGraph engine inside a visual state tracking spinner loop
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data patterns..."):
                try:
                    # Construct state dictionary input mirroring your model definition keys
                    inputs = {
                        "current_query": prompt,
                        "messages": st.session_state.messages
                    }
                    
                    # Pass state keys along with thread details to preserve long-term context tracking
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # Compute next node sequence modifications
                    output_state = app.invoke(inputs, config=config)
                    final_payload = output_state.get("final_response")

                    # 3. Stream compiled graphical assets out to the screen
                    display_agent_payload(final_payload)

                    # 4. Save assistant actions back to long-term memory stacks
                    if final_payload:
                        # Under Pydantic v2, AIMessage constructor strictly validates that content is a string.
                        # We store the response object in a custom attribute 'response_payload' so that msg.content remains
                        # a clean string. This prevents crashes when LangChain serializes history in subsequent turns.
                        content_str = getattr(final_payload, "answer", None) or getattr(final_payload, "message", "")
                        msg = AIMessage(content=content_str)
                        msg.response_payload = final_payload
                        st.session_state.messages.append(msg)
                    else:
                        fallback_msg = output_state["messages"][-1]
                        st.session_state.messages.append(fallback_msg)

                except Exception as ex:
                    logging.error(f"UI thread caught graph invocation crash: {str(ex)}", exc_info=True)
                    st.error(f"Workflow Exception Intercepted: {str(ex)}")


if __name__ == "__main__":
    main()