


# Talk Before You Judge: DUET – A Dual-LLM Framework for Improving Evaluation Stability
Current LLM-based evaluation systems often exhibit judgment instability, frequently changing their responses when prompted with follow-up queries that introduce no new information, such as “Are you sure?” To address this, we propose DUET (Deliberative Understanding for Evaluation Tasks) framework in which two large language models (LLMs) independently assess a task, exchange explanations, and revise their judgments through a single round of deliberation to improve decision stability. We systematically study response flipping across five prominent LLMs on 620 prompts spanning knowledge, mathematics, coding, and reasoning tasks. In addition to reducing flip rates, our approach significantly improves response stability, ensuring that models maintain consistent decisions under minimal challenge. Overall, our framework offers a dependable alternative to single-model evaluation and contributes to more trustworthy automated assessments.

---

## 🔧 What This Repository Contains

- ✅ Code for running **Traditional (single-LLM)** evaluation.
- 🤝 Code for our proposed **DUET (dual-LLM)** framework.
- 📊 Evaluation scripts for **flip rate** and **stability**
- 📁 Results and utility scripts.
- 📄 Instructions for reproducing all experiments from our paper.

---



To run the experiments, you will need valid API keys for the following models:

| Model     | Provider     | Access Link                            |
|-----------|--------------|----------------------------------------|
| GPT       | OpenAI       | https://platform.openai.com            |
| DeepSeek  | DeepSeek     | https://deepseek.com                   |
| Claude    | Anthropic    | https://console.anthropic.com          |
| LLaMA     | GROQ         | https://console.groq.com               |
| Gemma     | GROQ         | https://console.groq.com               |

> ⚠️ You must set your API keys as environment variables or edit the scripts accordingly.



