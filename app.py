import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# 设置 Streamlit 页面配置
st.set_page_config(page_title="Llama3 ChatBot with Search", page_icon="🦜")
st.title("🦜 Llama3: Chat with Search")

# 获取 API key 和 base URL（优先从 secrets 获取，其次从用户输入）
openai_api_key = st.sidebar.text_input("API Key", type="password")

api_model_name = st.sidebar.text_input("模型(可选)", value="rohan/Meta-Llama-3-70B-Instruct")

# 显示信息和链接
with st.sidebar:
    st.markdown("[llama3 API Key获取方式](https://nbid.bid/blog)")

# 初始化聊天历史和内存
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

# 重置聊天历史按钮
if len(msgs.messages) == 0 or st.sidebar.button("Reset Chat History"):
    msgs.clear()
    msgs.add_ai_message("请输入……")
    st.session_state.steps = {}

# 显示聊天历史，包括中间步骤
avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

# 处理用户输入
if prompt := st.chat_input(placeholder="请输入你的问题"):
    st.chat_message("user").write(prompt)

    # 检查 API Key
    if not openai_api_key:
        st.error("请添加您的 API key 以继续。")
        st.stop()
    
    # 初始化 LangChain 聊天机器人和工具

    llm = ChatOpenAI(
        model_name=api_model_name,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        streaming=True
    )
    tools = [DuckDuckGoSearchRun(name="Search")]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    # 执行聊天机器人并显示响应，包括错误处理
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        try:
            response = executor.invoke(prompt, cfg)
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
        except Exception as e:
            st.error(f"执行时发生错误：{e}")
