from classifier.classifier import (
    generate_classifications_from_csvs_with_clean_files,
    write_classes_to_final_csv,
    NUM_CASES,
    Response,
)

# Raw and clean files for the same dataset should have corresponding list positions here
# NB assumes column names will match across the two files
RAW_FILEPATHS = [
    "Gpt_llama-claude_coding.csv",
    "gpt_Llama-gpt4o_coding.csv",
    "GPT_llama-claude-Knowledge_raw.csv",
    "GPT_llama-claude-Maths_raw.csv",
]
CLEANED_FILEPATHS = [
    "Gpt_llama-claude_coding_clean.csv",
    "gpt_Llama-gpt4o_coding_clean.csv",
    "Gpt_llama-claude_knowledge_clean.csv",
    "Gpt_llama-claude_Maths_clean_1.csv",
]
# Any column matching this regex will be excluded during CSV import
EXCLUDE_COLUMNS_REGEX = "(Prompt1)|(Label)|(Unnamed: 1)|(.*_(i|I)nitial)"


def evaluate_classifier():
    classifications = generate_classifications_from_csvs_with_clean_files(
        RAW_FILEPATHS, CLEANED_FILEPATHS, EXCLUDE_COLUMNS_REGEX
    )

    accuracy = sum([c.correct for c in classifications]) / len(classifications)
    manual = sum([c.automatic_label == Response.Unsure for c in classifications]) / len(
        classifications
    )
    print(f"\nTotal LLM outputs classified: {len(classifications)}")
    print(
        f"Overall accuracy: {accuracy*100:.1f}% (including as incorrect the {manual*100:.1f}% marked to be manually classified)"
    )
    print("\n--------------- ACCURACY BY INPUT FILE ----------------")
    for filepath in RAW_FILEPATHS:
        accuracy = sum(
            [c.correct for c in classifications if c.source_filepath == filepath]
        ) / len([c for c in classifications if c.source_filepath == filepath])
        print(f"Accuracy for {filepath}: {accuracy*100:.1f}%")
    print("\n------ PROPORTION CLASSIFIED BY CLASSIFIER CASES ------")
    print("See comments in `classifier.py` for explanation of cases")
    for case in range(NUM_CASES):
        accuracy = len(
            [c for c in classifications if c.automatic_label_case == case]
        ) / len(classifications)
        print(f"Proportion classified using case {case}: {accuracy*100:.1f}%")
    print("\nSee log file and `classification_attempts.csv` for more information")

    # write_classes_to_final_csv(classifications)


def main():
    evaluate_classifier()


if __name__ == "__main__":
    main()
