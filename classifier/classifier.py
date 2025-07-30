import dataclasses
import csv
import logging
import re
from enum import Enum, auto, unique
from hashlib import md5
from typing import Optional

# the number of different approaches the classifier below tries;
# constant here used for aggregate statistics later on
NUM_CASES = 3

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler("app.log")
stream_handler.setLevel(logging.WARNING)
file_handler.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[stream_handler, file_handler],
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


@unique
class Response(Enum):
    A = auto()
    B = auto()
    Unsure = auto()


@dataclasses.dataclass()
class ResponsePattern:
    """Represents a pattern to match a common LLM output,
    e.g. `Response (A|B) is a better solution`, alongside
    the start index that the e.g. `ResponseA` substring will be
    in the output so it can be sliced out later on. NB pattern
    input should be in `no_punctuation` format (aside from the
    RegExp bits obviously)."""

    pattern: str
    res_start: int

    @property
    def res_end(self) -> int:
        return self.res_start + 9


@dataclasses.dataclass()
class ResponsePatternCustomOffset(ResponsePattern):
    """Same as response pattern but for other patterns than the standard
    "response A"/"response B", e.g. "first response"/"second response".
    Any use of this probably requires something to be added to the
    `string_to_response_enum` function if statement cases to match it."""

    offset: int

    @property
    def res_end(self) -> int:
        return self.res_start + self.offset


@dataclasses.dataclass
class ClassificationAttempt:
    """Represents an attempt by the automated classifier to assess which response
    the LLM is saying is the right one for a specific problem and LLM."""

    prompt: str  # from judgebench
    judgebench_label: str
    raw_llm_output: str
    manual_label: Optional[
        Response
    ]  # only needed if being tested with already manually labelled data
    automatic_label: Optional[Response]

    # included to allow stats to be generated on what proportion of the items are
    # being classified at each stage
    automatic_label_case: Optional[int]

    # included to allow stats to be aggregated by llm combination later on
    # should be the path to the raw file not the one with the `_clean` suffix
    source_filepath: Optional[str]
    source_column: Optional[str]
    source_line: Optional[int]  # line in csv file

    # only present to make it easier to find specific records in
    # output csv for debugging/exploration purposes
    id: str

    @property
    def correct(self) -> bool:
        """Return whether the automated classification agrees with the manual one
        or False if the automated classification hasn't been done yet. We
        don't want to be getting **any** wrong, so this method raises an
        exception if the automatic_label isn't just unknown but is actively wrong."""

        if (
            self.manual_label is not None
            and self.manual_label != self.automatic_label
            and (
                self.automatic_label == Response.A or self.automatic_label == Response.B
            )
        ):
            logging.error(f"Actively wrong automatic label set on {self}")
            # raise ValueError(f"Actively wrong automatic label set on {str(self)}")

        return self.manual_label == self.automatic_label

    def classify_llm_output(self) -> None:
        """Takes a single output from one of the LLMs (should be set in the `raw_llm_output` field) on a given question and assesses
        whether the model has chosen response 'A' or response 'B'. Returns None as it
        sets the answer on the class itself in the `automatic_label` field."""

        # remove all punctuation (including spaces) and make lower case
        # to increase performance of rule-based techniques + removes anything
        # in brackets with no spaces as this was sometimes causing issues
        no_punctuation = re.sub(
            r"[\W]+", "", re.sub(r"(\([a-zA-Z]*\))+", "", self.raw_llm_output)
        ).lower()
        # no_punctuation_with_newlines = re.sub(r"[^a-zA-Z0-9_\n]+", "", self.raw_llm_output).lower()

        # Case 0: the LLM output is just "Response A" or "Response B"
        # (ignoring punctuation/formatting) OR the very first characters
        # are just "Response A" or "Response B" (chatGPT likes to do this)
        if no_punctuation[:9] == "responsea" or no_punctuation[:9] == "responseb":
            logging.debug(f"classify_llm_output: Got Case 0 match")
            self.automatic_label = string_to_response_enum(no_punctuation[:9])
            self.automatic_label_case = 0
            return

        # Case 1: the LLM output contains a common statement
        # expressing its decision, which is matched by one of the
        # simple RegExps below
        definitive_patterns: list[ResponsePattern] = [
            # NB patterns numbered from 0 in the log file!
            ResponsePattern(r"explicitlychooseresponse(a|b)", 16),
            ResponsePattern(r"finalchoiceisresponse(a|b)", 13),
            ResponsePattern(r"finalanswerresponse(a|b)", 11),
            ResponsePattern(r"bestanswerisresponse(a|b)", 12),
            ResponsePattern(r"moreaccurateanswerisresponse(a|b)", 20),
            ResponsePattern(r"finalchoiceresponse(a|b)", 11),
            ResponsePattern(r"finalanswer(is|as)response(a|b)", 13),
            ResponsePattern(r"iwouldchooseresponse(a|b)", 12),
            ResponsePattern(r"ichooseresponse(a|b)", 7),
            ResponsePattern(r"thecorrectanswerisresponse(a|b)", 18),
            ResponsePattern(r"theanswerisresponse(a|b)", 11),
        ]
        suggestive_patterns: list[ResponsePattern] = [
            ResponsePattern(r"response(a|b)isthe(correct)?(and)?(better)?answer", 0),
            ResponsePattern(
                r"response(a|b)is(a)?(overall)?(slightly)?(ultimately)?(a|the)?better",
                0,
            ),
            ResponsePattern(r"response(a|b)(might)?bea(slightly)?betterchoice", 0),
            ResponsePattern(r"thebetterresponseisresponse(a|b)", 19),
            ResponsePattern(r"thebetteranswerisresponse(a|b)", 17),
            ResponsePattern(r"iwouldrecommendchoosingresponse(a|b)", 23),
            ResponsePattern(r"iwouldrecommendresponse(a|b)", 15),
            ResponsePattern(r"response(a|b)is(the)?moreaccurate", 0),
            ResponsePattern(r"correctresponseisresponse(a|b)", 17),
            ResponsePattern(r"moreaccurateresponseisresponse(a|b)", 22),
            ResponsePattern(r"response(a|b)ispreferable", 0),
            ResponsePattern(r"idoptforresponse(a|b)", 8),
            ResponsePattern(r"response(a|b)iscorrect", 0),
            ResponsePatternCustomOffset(
                r"(first|second)responseis(a)?(overall)?(slightly)?(ultimately)?(a|the)?better",
                0,
                14,
            ),
            ResponsePatternCustomOffset(r"betterresponseis(a|b)", 6, 11),
        ]

        # Using a voting approach to increase accuracy --
        # if there are some matches for Response A *and* some for Response B, we
        # just declare it as manually classified, to avoid a potential
        # incorrect automatic classification
        votes = {Response.A: 0, Response.B: 0}
        for i, p in enumerate(definitive_patterns):
            if (match := re.search(p.pattern, no_punctuation)) is not None:
                logging.debug(
                    f"classify_llm_output: Got Case 1 match with definitive pattern {i} on {self.id}"
                )
                votes[string_to_response_enum(match[0][p.res_start : p.res_end])] += 3
        for i, p in enumerate(suggestive_patterns):
            if (match := re.search(p.pattern, no_punctuation)) is not None:
                logging.debug(
                    f"classify_llm_output: Got Case 1 match with suggestive pattern {i} on {self.id}"
                )
                votes[string_to_response_enum(match[0][p.res_start : p.res_end])] += 1

        if (
            votes[Response.A] != 0
            and votes[Response.B] != 0
            and abs(votes[Response.A] - votes[Response.B]) >= 3
        ):
            logging.info(f"Voting mismatch found on {self.id}: {votes}")
        elif votes[Response.A] != 0 or votes[Response.B] != 0:
            self.automatic_label = (
                Response.A if votes[Response.A] > votes[Response.B] else Response.B
            )
            self.automatic_label_case = 1
            return

        # Case 2: manually classified by human (doesn't happen here,
        # see manual.py)

        self.automatic_label = Response.Unsure

        logging.info(self)

    def __str__(self) -> str:
        return f"ClassificationAttempt(id={self.id}, manual_label={self.manual_label}, automatic_label={self.automatic_label}, automatic_label_case={self.automatic_label_case}, source_filepath={self.source_filepath}, source_column={self.source_column}, source_line={self.source_line})"

    def __repr__(self) -> str:
        return str(self)


def string_to_response_enum(label: str) -> Response:
    """Convert string response to Response enum."""
    if (
        label == "Response A"
        or label == "responsea"
        or label == "responseisa"
        or re.fullmatch("firstresponse.", label)
    ):
        return Response.A
    elif (
        label == "Response B"
        or label == "responseb"
        or label == "responseisb"
        or label == "secondresponse"
    ):
        return Response.B
    else:
        raise ValueError(
            f"Label found not matching `Response A` or `Response B` (or no-punctuation, lower-case equivalents), got {label} instead"
        )


def generate_classifications_from_csvs(
    raw_filepaths: list[str],
    columns_to_exclude_regex: str,
) -> list[ClassificationAttempt]:
    """Returns list of ClassificationAttempt objects generated from all the CSV files provided.
    A ClassificationAttempt object (without an `automatic_label`) is generated
    for each relevant cell across the input files and is then classified.
    Columns with names matching the `columns_to_exclude_regex` will not be excluded."""

    classifications = []
    for raw_filepath in raw_filepaths:
        with open(raw_filepath) as raw_file:
            raw_reader = csv.DictReader(raw_file, delimiter=",")
            line = 2
            for raw_row in raw_reader:
                relevant_columns = [
                    col
                    for col in raw_row.keys()
                    if not re.match(columns_to_exclude_regex, col)
                ]
                for col in relevant_columns:
                    classifications.append(
                        ClassificationAttempt(
                            raw_row["Prompt1"],
                            raw_row["Label"],
                            raw_row[col],
                            None,
                            None,
                            None,
                            raw_filepath,
                            col,
                            line,
                            md5(raw_row[col].encode("utf-8")).hexdigest(),
                        )
                    )
                line += 1

    for classification in classifications:
        classification.classify_llm_output()

    write_classes_to_debug_csv("classification_attempts.csv", classifications)

    return classifications


def generate_classifications_from_csvs_with_clean_files(
    raw_filepaths: list[str],
    cleaned_filepaths: list[str],
    columns_to_exclude_regex: str,
) -> list[ClassificationAttempt]:
    """Returns list of ClassificationAttempt objects generated from all the CSV files provided.
    Raw and clean files for the same dataset should have corresponding list positions
    in the input lists. A ClassificationAttempt object (without an `automatic_label`) is generated
    for each relevant cell across the input files and is then classified.
    Columns with names matching the `columns_to_exclude_regex` will not be excluded."""

    classifications = []
    for raw_filepath, cleaned_filepath in zip(raw_filepaths, cleaned_filepaths):
        with open(raw_filepath) as raw_file, open(cleaned_filepath) as clean_file:
            raw_reader = csv.DictReader(raw_file, delimiter=",")
            clean_reader = csv.DictReader(clean_file, delimiter=",")
            line = 2
            for raw_row, clean_row in zip(raw_reader, clean_reader):
                assert raw_row["Prompt1"] == clean_row["Prompt1"]
                relevant_columns = set(raw_row.keys()).union(clean_row.keys())
                relevant_columns = relevant_columns.difference(
                    col
                    for col in relevant_columns
                    if re.match(columns_to_exclude_regex, col)
                )
                for col in relevant_columns:
                    classifications.append(
                        ClassificationAttempt(
                            raw_row["Prompt1"],
                            "",  # these csvs tend not to have judgebench labels
                            raw_row[col],
                            string_to_response_enum(clean_row[col]),
                            None,
                            None,
                            raw_filepath,
                            col,
                            line,
                            md5(raw_row[col].encode("utf-8")).hexdigest(),
                        )
                    )
                line += 1

    for classification in classifications:
        classification.classify_llm_output()

    write_classes_to_debug_csv("classification_attempts.csv", classifications)

    return classifications


def write_classes_to_debug_csv(
    csv_filepath: str, objects: list[ClassificationAttempt]
) -> None:
    """Takes a list of objects, here classification attempts, and writes them to a
    CSV file for easier analysis."""
    with open(csv_filepath, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow((*objects[0].__annotations__.keys(), "correct"))

        # Write data
        for instance in objects:
            writer.writerow((*instance.__dict__.values(), instance.correct))


def write_classes_to_final_csv(attempts: list[ClassificationAttempt]) -> None:
    """Takes a list of classification attempts, and writes them to an output CSV
    file automatically (file name given by their source file name + "_clean_gen")."""

    # group by source csv
    source_file_to_attempts: dict[str, list[ClassificationAttempt]] = {}
    for attempt in attempts:
        source_file_to_attempts.setdefault(attempt.source_filepath, []).append(attempt)

    # group by source csv AND THEN prompt within it
    source_file_to_grouped_attempts: dict[
        str, dict[str, list[ClassificationAttempt]]
    ] = {filepath: {} for filepath in source_file_to_attempts}
    for filepath in source_file_to_attempts:
        for attempt in source_file_to_attempts[filepath]:
            source_file_to_grouped_attempts[filepath].setdefault(
                attempt.prompt, []
            ).append(attempt)

    # actually write to CSVs
    for raw_path in source_file_to_attempts:
        clean_path = re.sub(r"\.csv", "_clean_gen.csv", raw_path)
        with open(raw_path) as raw_f, open(clean_path, mode="w", newline="") as clean_f:
            reader = csv.DictReader(raw_f, delimiter=",")
            writer = csv.writer(clean_f)

            if reader.fieldnames is None:
                raise ValueError(
                    f"Something went wrong with CSV export: raw file reader.fieldnames is None for raw file {raw_path}"
                )

            # write header
            writer.writerow(col for col in reader.fieldnames if col != "Unnamed: 1")
            assert reader.fieldnames[:2] == ["Prompt1", "Label"]

            # write rows
            for row in reader:
                output_row = [
                    source_file_to_grouped_attempts[raw_path][row["Prompt1"]][0].prompt,
                    source_file_to_grouped_attempts[raw_path][row["Prompt1"]][
                        0
                    ].judgebench_label,
                ]
                for col in list(row.keys())[2:]:
                    if re.match(r".*_Initial", col):
                        output_row.append(row[col])
                    else:
                        for attempt in source_file_to_grouped_attempts[raw_path][
                            row["Prompt1"]
                        ]:
                            if attempt.source_column == col:
                                string_repr = (
                                    "Response A"
                                    if attempt.automatic_label == Response.A
                                    else (
                                        "Response B"
                                        if attempt.automatic_label == Response.B
                                        else "UNSURE"
                                    )
                                )
                                output_row.append(string_repr)
                                break
                        else:
                            logging.error(
                                f"Problem generating clean CSV -- {col} not found"
                            )
                writer.writerow(output_row)
