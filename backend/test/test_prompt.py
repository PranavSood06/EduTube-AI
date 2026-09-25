from app.services.rag.prompt import prompts


def test_youtube_chat_prompt_includes_context_and_question():
    rendered = prompts.YTchatPrompt().format_messages(
        context="Newton's first law describes inertia.",
        question="What is inertia?",
    )

    content = rendered[0].content
    assert "Newton's first law describes inertia." in content
    assert "What is inertia?" in content
    assert "Do not make up facts" in content
