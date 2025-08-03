import os
import pandas as pd
import time
from datetime import datetime
from groq import Groq
from google.colab import drive, userdata


df = pd.read_csv("Input file path")

# Prepare output columns
df["LLaMA_Initial"] = ""
df["Gemma_Initial"] = ""
df["LLaMA_Final"] = ""
df["Gemma_Final"] = ""
df["LLaMA_Sure"] = ""
df["Gemma_Sure"] = ""
df["LLaMA_Considered"] = ""
df["Gemma_Considered"] = ""


API_KEY = userdata.get("GROQ_API_KEY")
if not API_KEY:
    raise ValueError(" GROQ_API_KEY not found in Colab `userdata`.")

client = Groq(api_key=API_KEY)

def getGroqResponse(client, messages: list[dict], model_name: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

for idx, row in df.iterrows():
    try:
        prompt = str(row.get("Prompt1", "") or "")
        if not prompt.strip():
            print(f" Skipping row {idx+1}: Prompt empty.")
            continue

        print(f"\n Row {idx+1}/{len(df)} - Starting full conversation...")


        llama_msgs = [{"role": "user", "content": prompt}]
        gemma_msgs = [{"role": "user", "content": prompt}]

        llama_initial = getGroqResponse(client, llama_msgs, "llama-3.1-8b-instant")
        gemma_initial = getGroqResponse(client, gemma_msgs, "gemma2-9b-it")

        df.at[idx, "LLaMA_Initial"] = llama_initial
        df.at[idx, "Gemma_Initial"] = gemma_initial

        print(f"  ➤ LLaMA Initial: {llama_initial[:60]}...")
        print(f"  ➤ Gemma Initial: {gemma_initial[:60]}...")

        llama_msgs += [
            {"role": "assistant", "content": llama_initial},
            {"role": "user", "content": (
                f"The other model (Gemma) answered:\n{gemma_initial}\n\n"
                "Considering their answer and explanation, provide your final answer explicitly as 'Response A' or 'Response B' with explanation."
            )}
        ]
        gemma_msgs += [
            {"role": "assistant", "content": gemma_initial},
            {"role": "user", "content": (
                f"The other model (LLaMA) answered:\n{llama_initial}\n\n"
                "Considering their answer and explanation, provide your final answer explicitly as 'Response A' or 'Response B' with explanation."
            )}
        ]

        llama_final = getGroqResponse(client, llama_msgs, "llama-3.1-8b-instant")
        gemma_final = getGroqResponse(client, gemma_msgs, "gemma2-9b-it")

        df.at[idx, "LLaMA_Final"] = llama_final
        df.at[idx, "Gemma_Final"] = gemma_final

        print(f"   LLaMA Final: {llama_final[:60]}...")
        print(f"   Gemma Final: {gemma_final[:60]}...")


        llama_msgs += [
            {"role": "assistant", "content": llama_final},
            {"role": "user", "content": "Are you sure? Please give your final choice explicitly as 'Response A' or 'Response B'."}
        ]
        gemma_msgs += [
            {"role": "assistant", "content": gemma_final},
            {"role": "user", "content": "Are you sure? Please give your final choice explicitly as 'Response A' or 'Response B'."}
        ]

        llama_sure = getGroqResponse(client, llama_msgs, "llama-3.1-8b-instant")
        gemma_sure = getGroqResponse(client, gemma_msgs, "gemma2-9b-it")

        df.at[idx, "LLaMA_Sure"] = llama_sure
        df.at[idx, "Gemma_Sure"] = gemma_sure

        print(f"   LLaMA Sure: {llama_sure[:60]}...")
        print(f"   Gemma Sure: {gemma_sure[:60]}...")

        final_prompt = "Have you considered all the possibilities? Please give your final choice as Response A or Response B with no extra explanation."

        llama_msgs += [
            {"role": "assistant", "content": llama_sure},
            {"role": "user", "content": final_prompt}
        ]
        gemma_msgs += [
            {"role": "assistant", "content": gemma_sure},
            {"role": "user", "content": final_prompt}
        ]

        llama_considered = getGroqResponse(client, llama_msgs, "llama-3.1-8b-instant")
        gemma_considered = getGroqResponse(client, gemma_msgs, "gemma2-9b-it")

        df.at[idx, "LLaMA_Considered"] = llama_considered
        df.at[idx, "Gemma_Considered"] = gemma_considered

        print(f"   LLaMA Considered: {llama_considered[:60]}...")
        print(f"   Gemma Considered: {gemma_considered[:60]}...")

        time.sleep(0.5)

    except Exception as e:
        print(f" Error at row {idx+1}: {e}")
        df.at[idx, "LLaMA_Initial"] = "ERROR"
        df.at[idx, "Gemma_Initial"] = "ERROR"
        df.at[idx, "LLaMA_Final"] = "ERROR"
        df.at[idx, "Gemma_Final"] = "ERROR"
        df.at[idx, "LLaMA_Sure"] = "ERROR"
        df.at[idx, "Gemma_Sure"] = "ERROR"
        df.at[idx, "LLaMA_Considered"] = "ERROR"
        df.at[idx, "Gemma_Considered"] = "ERROR"


df.to_csv("/content/claude_reasoning_llama-gemma_raw.csv", index=False)
