from decision_module import use_cloud_llm

def answer_question(summary_text, question):
    if use_cloud_llm(summary_text + question):
        return cloud_answer(summary_text, question)
    return local_answer(summary_text, question)
