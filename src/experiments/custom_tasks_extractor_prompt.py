# NOTE: This file is long because lighteval copies the file to a temporary directory
# before running, therefore relative imports cannot be used
# TODO: Move most of the code into a package

from typing import List

import lighteval.tasks.default_prompts as prompt

from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import (
    multilingual_extractive_match_metric,
    IndicesExtractionConfig,
    IndicesExtractionConfig
)

import lighteval.tasks.extended.ifeval

import ast

import random


metric_name = 'letter'


LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


COT_QUESTION = """
    {Instruction}
    On the very last line, write exactly "Answer: $LETTER" (e.g. "Answer: B"), with no extra punctuation, no lowercase, no *, and no trailing spaces.
    Think step by step, showing your reasoning.
    Question: "{Question}"
    """

reasoning_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


def create_evaluation_task(
    namespace: str,
    category: str,
    hf_subset: str,
    prompt_function,
    hf_repo: str,
    evaluation_splits: List[str] = ['train'],
    suite: List[str] = ['community'],
    hf_avail_splits: List[str] = ['train'],
    few_shots_split: str = 'validation',
    generation_size=None, # use model's default "max_position_embeddings"
    stop_sequence=['<|im_start|>', 'Question: ', 'You are a helpful assistant']) -> LightevalTaskConfig:    
    name = f'{namespace}_{metric_name}:{category}'
    task = LightevalTaskConfig(
        name=name,
        prompt_function=prompt_function,
        suite=suite,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select="sequential",
        metric=[reasoning_metric],
        generation_size=generation_size,
        trust_dataset=True,
        stop_sequence=stop_sequence
    )
    
    return task


def create_question_with_known_choices(instruction: str, question: str, choices: List[str]):
    query = [COT_QUESTION.format(Instruction=instruction, Question=question)]

    for i, choice in enumerate(choices):
        query.append(f'{LETTER_INDICES[i]}) {choice}')
    
    return '\n'.join(query).strip()


def create_question_without_known_choices(instruction: str, question: str):
    query = [COT_QUESTION.format(Instruction=instruction, Question=question)]

    return '\n'.join(query).strip()


def bbh_with_cot_with_known_choices(line, task_name: str, instruction: str, choices: List[str]):
    gold_tf = line["target"]
    gold_index_tf = choices.index(gold_tf)
    gold_letter = LETTER_INDICES[:len(choices)][gold_index_tf]
    query = create_question_with_known_choices(instruction=instruction, question=line['input'], choices=choices)

    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=letter_list.index(gold_letter),
        instruction=query,
    )

def bbh_with_cot_without_known_choices(line, task_name: str, instruction: str, n_choices: int, use_numeric: bool = False):
    choices = [f"({c})" for c in LETTER_INDICES[:n_choices]]
    if use_numeric:
        choices = [f"{c}" for c in range(1, n_choices + 1)]

    gold_tf = line["target"]
    gold_index_tf = choices.index(gold_tf)
    gold_letter = LETTER_INDICES[:n_choices][gold_index_tf]
    query = create_question_without_known_choices(instruction=instruction, question=line['input'])

    letter_list = [c for c in LETTER_INDICES[:n_choices]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index= letter_list.index(gold_letter),
        instruction=query,
    )


def bbh_boolean_expressions(line, task_name: str = None):
    instruction = "Evaluate the result of a random Boolean expression.\n\n"
    choices = ["False", "True"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_causal_judgement(line, task_name: str = None):
    instruction = "Answer questions about causal attribution.\n\n"
    choices = ["Yes", "No"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_date_understanding(line, task_name: str = None):
    instruction = "Infer the date from context.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6)


def bbh_disambiguation_qa(line, task_name: str = None):
    instruction = "Clarify the meaning of sentences with ambiguous pronouns.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3)


def bbh_dyck_languages(line, task_name: str = None):
    instruction = "Correctly close a Dyck-n word.\n\n"
    choices = [line["target"]]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_formal_fallacies(line, task_name: str = None):
    instruction = "Distinguish deductively valid arguments from formal fallacies.\n\n"
    choices = ["valid", "invalid"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_geometric_shapes(line, task_name: str = None):
    instruction = "Name geometric shapes from their SVG paths.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=11)


def bbh_hyperbaton(line, task_name: str = None):
    instruction = "Order adjectives correctly in English sentences.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=2)


def bbh_logical_deduction_five_objects(line, task_name: str = None):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5)


def bbh_logical_deduction_seven_objects(line, task_name: str = None):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=7)


def bbh_logical_deduction_three_objects(line, task_name: str = None):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3)


def bbh_movie_recommendation(line, task_name: str = None):
    if line["target"] == "Monsters, Inc":  # this line is not correctly formatted
        print(
            "One sample removed from task bbh:movie_recommendation because its line is incorrectly formatted."
        )
        return []
    instruction = "Recommend movies similar to the given list of movies.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6)


def bbh_multistep_arithmetic_two(line, task_name: str = None):
    instruction = "Solve multi-step arithmetic problems.\n\n"
    choices = [line["target"]]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_navigate(line, task_name: str = None):
    instruction = (
        "Given a series of navigation instructions, determine whether one would end up back at the starting point.\n\n"
    )
    choices = ["Yes", "No"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_object_counting(line, task_name: str = None):
    instruction = "Questions that involve enumerating objects and asking the model to count them.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=19, use_numeric=True)


def bbh_penguins_in_a_table(line, task_name: str = None):
    instruction = "Answer questions about a table of penguins and their attributes.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5)


def bbh_reasoning_about_colored_objects(line, task_name: str = None):
    instruction = "Answer extremely simple questions about the colors of objects on a surface.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=18)


def bbh_ruin_names(line, task_name: str = None):
    if line["target"] in ["dearth, wind, & fire", "rita, sue and bob poo"]:  # line not correctly formatted
        print("One sample removed from task bbh:ruin_names because its line is incorrectly formatted.")
        return []
    instruction = "Select the humorous edit that 'ruins' the input movie or musical artist name.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6)


def bbh_salient_translation_error_detection(line, task_name: str = None):
    instruction = "Detect the type of error in an English translation of a German source sentence.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6)


def bbh_snarks(line, task_name: str = None):
    instruction = 'Determine which of two sentences is sarcastic.\n\nAccording to Cambridge University Dictionary, sarcasm is "the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone\'s feelings or to criticize something in a humorous way." Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.\n\n'
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=2)


def bbh_sports_understanding(line, task_name: str = None):
    instruction = "Determine whether an artificially constructed sentence relating to sports is plausible or not.\n\n"
    choices = ["yes", "no"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_temporal_sequences(line, task_name: str = None):
    instruction = "Task description: Answer questions about which times certain events could have occurred.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=4)


def bbh_tracking_shuffled_objects_five_objects(line, task_name: str = None):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5)


def bbh_tracking_shuffled_objects_seven_objects(line, task_name: str = None):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=7)


def bbh_tracking_shuffled_objects_three_objects(line, task_name: str = None):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3)


def bbh_web_of_lies(line, task_name: str = None):
    instruction = "Evaluate a random boolean function expressed as a word problem.\n\n"
    choices = ["Yes", "No"]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def bbh_word_sorting(line, task_name: str = None):
    instruction = "Sort a list of words.\n\n"
    choices = [line["target"]]
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices)


def create_bbh_table():
    return [
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='boolean_expressions',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="boolean_expressions",
            prompt_function=bbh_boolean_expressions,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='causal_judgment', # spelling changed to match "harness|bbh:causal_judgment"
            hf_repo='lighteval/big_bench_hard',
            hf_subset="causal_judgement",
            prompt_function=bbh_causal_judgement,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='date_understanding',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="date_understanding",
            prompt_function=bbh_date_understanding,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='disambiguation_qa',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="disambiguation_qa",
            prompt_function=bbh_disambiguation_qa,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='dyck_languages',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="dyck_languages",
            prompt_function=bbh_dyck_languages,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='formal_fallacies',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="formal_fallacies",
            prompt_function=bbh_formal_fallacies,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='geometric_shapes',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="geometric_shapes",
            prompt_function=bbh_geometric_shapes,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='hyperbaton',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="hyperbaton",
            prompt_function=bbh_hyperbaton,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='logical_deduction_five_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="logical_deduction_five_objects",
            prompt_function=bbh_logical_deduction_five_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='logical_deduction_seven_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="logical_deduction_seven_objects",
            prompt_function=bbh_logical_deduction_seven_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='logical_deduction_three_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="logical_deduction_three_objects",
            prompt_function=bbh_logical_deduction_three_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='movie_recommendation',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="movie_recommendation",
            prompt_function=bbh_movie_recommendation,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='multistep_arithmetic_two',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="multistep_arithmetic_two",
            prompt_function=bbh_multistep_arithmetic_two,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='navigate',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="navigate",
            prompt_function=bbh_navigate,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='object_counting',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="object_counting",
            prompt_function=bbh_object_counting,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='penguins_in_a_table',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="penguins_in_a_table",
            prompt_function=bbh_penguins_in_a_table,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='reasoning_about_colored_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="reasoning_about_colored_objects",
            prompt_function=bbh_reasoning_about_colored_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='ruin_names',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="ruin_names",
            prompt_function=bbh_ruin_names,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='salient_translation_error_detection',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="salient_translation_error_detection",
            prompt_function=bbh_salient_translation_error_detection,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='snarks',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="snarks",
            prompt_function=bbh_snarks,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='sports_understanding',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="sports_understanding",
            prompt_function=bbh_sports_understanding,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='temporal_sequences',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="temporal_sequences",
            prompt_function=bbh_temporal_sequences,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='tracking_shuffled_objects_five_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="tracking_shuffled_objects_five_objects",
            prompt_function=bbh_tracking_shuffled_objects_five_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='tracking_shuffled_objects_seven_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="tracking_shuffled_objects_seven_objects",
            prompt_function=bbh_tracking_shuffled_objects_seven_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='tracking_shuffled_objects_three_objects',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="tracking_shuffled_objects_three_objects",
            prompt_function=bbh_tracking_shuffled_objects_three_objects,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='web_of_lies',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="web_of_lies",
            prompt_function=bbh_web_of_lies,
        ),
        create_evaluation_task(
            namespace='bbh_reasoning',
            category='word_sorting',
            hf_repo='lighteval/big_bench_hard',
            hf_subset="word_sorting",
            prompt_function=bbh_word_sorting,
        )
    ]


def gpqa_with_cot(line, task_name: str = None):
    instruction = "Answer the following multiple choice question."

    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    query = create_question_with_known_choices(
        instruction=instruction,
        question=line["Question"],
        choices=choices
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=query,
    )


def create_gpqa_table():
    return [
        create_evaluation_task(
            namespace="gpqa_reasoning",
            category="diamond",
            hf_repo='Idavidrein/gpqa',
            hf_subset='gpqa_diamond',
            prompt_function=gpqa_with_cot
        )
    ]


def mmlu_with_cot(line, topic, task_name: str = None):
    instruction = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}."
    question = line["question"]
    choices = line["choices"]
    
    query = create_question_with_known_choices(instruction=instruction,
                                               question=question,
                                               choices=choices)

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_ix,
        instruction=instruction,
    )


def mmlu_abstract_algebra(line, task_name: str = None):
    return mmlu_with_cot(line, "abstract_algebra", task_name)


def mmlu_anatomy(line, task_name: str = None):
    return mmlu_with_cot(line, "anatomy", task_name)


def mmlu_astronomy(line, task_name: str = None):
    return mmlu_with_cot(line, "astronomy", task_name)


def mmlu_business_ethics(line, task_name: str = None):
    return mmlu_with_cot(line, "business_ethics", task_name)


def mmlu_clinical_knowledge(line, task_name: str = None):
    return mmlu_with_cot(line, "clinical_knowledge", task_name)


def mmlu_college_biology(line, task_name: str = None):
    return mmlu_with_cot(line, "college_biology", task_name)


def mmlu_college_chemistry(line, task_name: str = None):
    return mmlu_with_cot(line, "college_chemistry", task_name)


def mmlu_college_computer_science(line, task_name: str = None):
    return mmlu_with_cot(line, "college_computer_science", task_name)


def mmlu_college_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, "college_mathematics", task_name)


def mmlu_college_medicine(line, task_name: str = None):
    return mmlu_with_cot(line, "college_medicine", task_name)


def mmlu_college_physics(line, task_name: str = None):
    return mmlu_with_cot(line, "college_physics", task_name)


def mmlu_computer_security(line, task_name: str = None):
    return mmlu_with_cot(line, "computer_security", task_name)


def mmlu_conceptual_physics(line, task_name: str = None):
    return mmlu_with_cot(line, "conceptual_physics", task_name)


def mmlu_econometrics(line, task_name: str = None):
    return mmlu_with_cot(line, "econometrics", task_name)


def mmlu_electrical_engineering(line, task_name: str = None):
    return mmlu_with_cot(line, "electrical_engineering", task_name)


def mmlu_elementary_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, "elementary_mathematics", task_name)


def mmlu_formal_logic(line, task_name: str = None):
    return mmlu_with_cot(line, "formal_logic", task_name)


def mmlu_global_facts(line, task_name: str = None):
    return mmlu_with_cot(line, "global_facts", task_name)


def mmlu_high_school_biology(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_biology", task_name)


def mmlu_high_school_chemistry(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_chemistry", task_name)


def mmlu_high_school_computer_science(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_computer_science", task_name)


def mmlu_high_school_european_history(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_european_history", task_name)


def mmlu_high_school_geography(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_geography", task_name)


def mmlu_high_school_government_and_politics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_government_and_politics", task_name)


def mmlu_high_school_macroeconomics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_macroeconomics", task_name)


def mmlu_high_school_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_mathematics", task_name)


def mmlu_high_school_microeconomics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_microeconomics", task_name)


def mmlu_high_school_physics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_physics", task_name)


def mmlu_high_school_psychology(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_psychology", task_name)


def mmlu_high_school_statistics(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_statistics", task_name)


def mmlu_high_school_us_history(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_us_history", task_name)


def mmlu_high_school_world_history(line, task_name: str = None):
    return mmlu_with_cot(line, "high_school_world_history", task_name)


def mmlu_human_aging(line, task_name: str = None):
    return mmlu_with_cot(line, "human_aging", task_name)


def mmlu_human_sexuality(line, task_name: str = None):
    return mmlu_with_cot(line, "human_sexuality", task_name)


def mmlu_international_law(line, task_name: str = None):
    return mmlu_with_cot(line, "international_law", task_name)


def mmlu_jurisprudence(line, task_name: str = None):
    return mmlu_with_cot(line, "jurisprudence", task_name)


def mmlu_logical_fallacies(line, task_name: str = None):
    return mmlu_with_cot(line, "logical_fallacies", task_name)


def mmlu_machine_learning(line, task_name: str = None):
    return mmlu_with_cot(line, "machine_learning", task_name)


def mmlu_management(line, task_name: str = None):
    return mmlu_with_cot(line, "management", task_name)


def mmlu_marketing(line, task_name: str = None):
    return mmlu_with_cot(line, "marketing", task_name)


def mmlu_medical_genetics(line, task_name: str = None):
    return mmlu_with_cot(line, "medical_genetics", task_name)


def mmlu_miscellaneous(line, task_name: str = None):
    return mmlu_with_cot(line, "miscellaneous", task_name)


def mmlu_moral_disputes(line, task_name: str = None):
    return mmlu_with_cot(line, "moral_disputes", task_name)


def mmlu_moral_scenarios(line, task_name: str = None):
    return mmlu_with_cot(line, "moral_scenarios", task_name)


def mmlu_nutrition(line, task_name: str = None):
    return mmlu_with_cot(line, "nutrition", task_name)


def mmlu_philosophy(line, task_name: str = None):
    return mmlu_with_cot(line, "philosophy", task_name)


def mmlu_prehistory(line, task_name: str = None):
    return mmlu_with_cot(line, "prehistory", task_name)


def mmlu_professional_accounting(line, task_name: str = None):
    return mmlu_with_cot(line, "professional_accounting", task_name)


def mmlu_professional_law(line, task_name: str = None):
    return mmlu_with_cot(line, "professional_law", task_name)


def mmlu_professional_medicine(line, task_name: str = None):
    return mmlu_with_cot(line, "professional_medicine", task_name)


def mmlu_professional_psychology(line, task_name: str = None):
    return mmlu_with_cot(line, "professional_psychology", task_name)


def mmlu_public_relations(line, task_name: str = None):
    return mmlu_with_cot(line, "public_relations", task_name)


def mmlu_security_studies(line, task_name: str = None):
    return mmlu_with_cot(line, "security_studies", task_name)


def mmlu_sociology(line, task_name: str = None):
    return mmlu_with_cot(line, "sociology", task_name)


def mmlu_us_foreign_policy(line, task_name: str = None):
    return mmlu_with_cot(line, "us_foreign_policy", task_name)


def mmlu_virology(line, task_name: str = None):
    return mmlu_with_cot(line, "virology", task_name)


def mmlu_world_religions(line, task_name: str = None):
    return mmlu_with_cot(line, "world_religions", task_name)


def create_mmlu_table():
    return [
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='abstract_algebra',
            hf_repo='lighteval/mmlu',
            hf_subset="abstract_algebra",
            evaluation_splits=['test'],
            prompt_function=mmlu_abstract_algebra,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='anatomy',
            hf_repo='lighteval/mmlu',
            hf_subset='anatomy',
            evaluation_splits=['test'],
            prompt_function=mmlu_anatomy,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='astronomy',
            hf_repo='lighteval/mmlu',
            hf_subset='astronomy',
            evaluation_splits=['test'],
            prompt_function=mmlu_astronomy,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='business_ethics',
            hf_repo='lighteval/mmlu',
            hf_subset='business_ethics',
            evaluation_splits=['test'],
            prompt_function=mmlu_business_ethics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='clinical_knowledge',
            hf_repo='lighteval/mmlu',
            hf_subset='clinical_knowledge',
            evaluation_splits=['test'],
            prompt_function=mmlu_clinical_knowledge,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_biology',
            hf_repo='lighteval/mmlu',
            hf_subset='college_biology',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_biology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_chemistry',
            hf_repo='lighteval/mmlu',
            hf_subset='college_chemistry',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_chemistry,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_computer_science',
            hf_repo='lighteval/mmlu',
            hf_subset='college_computer_science',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_computer_science,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_mathematics',
            hf_repo='lighteval/mmlu',
            hf_subset='college_mathematics',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_mathematics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_medicine',
            hf_repo='lighteval/mmlu',
            hf_subset='college_medicine',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_medicine,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='college_physics',
            hf_repo='lighteval/mmlu',
            hf_subset='college_physics',
            evaluation_splits=['test'],
            prompt_function=mmlu_college_physics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='computer_security',
            hf_repo='lighteval/mmlu',
            hf_subset='computer_security',
            evaluation_splits=['test'],
            prompt_function=mmlu_computer_security,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='conceptual_physics',
            hf_repo='lighteval/mmlu',
            hf_subset='conceptual_physics',
            evaluation_splits=['test'],
            prompt_function=mmlu_conceptual_physics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='econometrics',
            hf_repo='lighteval/mmlu',
            hf_subset='econometrics',
            evaluation_splits=['test'],
            prompt_function=mmlu_econometrics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='electrical_engineering',
            hf_repo='lighteval/mmlu',
            hf_subset='electrical_engineering',
            evaluation_splits=['test'],
            prompt_function=mmlu_electrical_engineering,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='elementary_mathematics',
            hf_repo='lighteval/mmlu',
            hf_subset='elementary_mathematics',
            evaluation_splits=['test'],
            prompt_function=mmlu_elementary_mathematics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='formal_logic',
            hf_repo='lighteval/mmlu',
            hf_subset='formal_logic',
            evaluation_splits=['test'],
            prompt_function=mmlu_formal_logic,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='global_facts',
            hf_repo='lighteval/mmlu',
            hf_subset='global_facts',
            evaluation_splits=['test'],
            prompt_function=mmlu_global_facts,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_biology',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_biology',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_biology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_chemistry',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_chemistry',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_chemistry,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_computer_science',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_computer_science',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_computer_science,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_european_history',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_european_history',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_european_history,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_geography',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_geography',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_geography,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_government_and_politics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_government_and_politics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_government_and_politics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_macroeconomics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_macroeconomics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_macroeconomics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_mathematics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_mathematics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_mathematics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_microeconomics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_microeconomics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_microeconomics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_physics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_physics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_physics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_psychology',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_psychology',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_psychology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_statistics',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_statistics',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_statistics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_us_history',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_us_history',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_us_history,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='high_school_world_history',
            hf_repo='lighteval/mmlu',
            hf_subset='high_school_world_history',
            evaluation_splits=['test'],
            prompt_function=mmlu_high_school_world_history,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='human_aging',
            hf_repo='lighteval/mmlu',
            hf_subset='human_aging',
            evaluation_splits=['test'],
            prompt_function=mmlu_human_aging,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='human_sexuality',
            hf_repo='lighteval/mmlu',
            hf_subset='human_sexuality',
            evaluation_splits=['test'],
            prompt_function=mmlu_human_sexuality,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='international_law',
            hf_repo='lighteval/mmlu',
            hf_subset='international_law',
            evaluation_splits=['test'],
            prompt_function=mmlu_international_law,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='jurisprudence',
            hf_repo='lighteval/mmlu',
            hf_subset='jurisprudence',
            evaluation_splits=['test'],
            prompt_function=mmlu_jurisprudence,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='logical_fallacies',
            hf_repo='lighteval/mmlu',
            hf_subset='logical_fallacies',
            evaluation_splits=['test'],
            prompt_function=mmlu_logical_fallacies,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='machine_learning',
            hf_repo='lighteval/mmlu',
            hf_subset='machine_learning',
            evaluation_splits=['test'],
            prompt_function=mmlu_machine_learning,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='management',
            hf_repo='lighteval/mmlu',
            hf_subset='management',
            evaluation_splits=['test'],
            prompt_function=mmlu_management,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='marketing',
            hf_repo='lighteval/mmlu',
            hf_subset='marketing',
            evaluation_splits=['test'],
            prompt_function=mmlu_marketing,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='medical_genetics',
            hf_repo='lighteval/mmlu',
            hf_subset='medical_genetics',
            evaluation_splits=['test'],
            prompt_function=mmlu_medical_genetics,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='miscellaneous',
            hf_repo='lighteval/mmlu',
            hf_subset='miscellaneous',
            evaluation_splits=['test'],
            prompt_function=mmlu_miscellaneous,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='moral_disputes',
            hf_repo='lighteval/mmlu',
            hf_subset='moral_disputes',
            evaluation_splits=['test'],
            prompt_function=mmlu_moral_disputes,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='moral_scenarios',
            hf_repo='lighteval/mmlu',
            hf_subset='moral_scenarios',
            evaluation_splits=['test'],
            prompt_function=mmlu_moral_scenarios,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='nutrition',
            hf_repo='lighteval/mmlu',
            hf_subset='nutrition',
            evaluation_splits=['test'],
            prompt_function=mmlu_nutrition,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='philosophy',
            hf_repo='lighteval/mmlu',
            hf_subset='philosophy',
            evaluation_splits=['test'],
            prompt_function=mmlu_philosophy,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='prehistory',
            hf_repo='lighteval/mmlu',
            hf_subset='prehistory',
            evaluation_splits=['test'],
            prompt_function=mmlu_prehistory,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='professional_accounting',
            hf_repo='lighteval/mmlu',
            hf_subset='professional_accounting',
            evaluation_splits=['test'],
            prompt_function=mmlu_professional_accounting,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='professional_law',
            hf_repo='lighteval/mmlu',
            hf_subset='professional_law',
            evaluation_splits=['test'],
            prompt_function=mmlu_professional_law,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='professional_medicine',
            hf_repo='lighteval/mmlu',
            hf_subset='professional_medicine',
            evaluation_splits=['test'],
            prompt_function=mmlu_professional_medicine,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='professional_psychology',
            hf_repo='lighteval/mmlu',
            hf_subset='professional_psychology',
            evaluation_splits=['test'],
            prompt_function=mmlu_professional_psychology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='public_relations',
            hf_repo='lighteval/mmlu',
            hf_subset='public_relations',
            evaluation_splits=['test'],
            prompt_function=mmlu_public_relations,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='security_studies',
            hf_repo='lighteval/mmlu',
            hf_subset='security_studies',
            evaluation_splits=['test'],
            prompt_function=mmlu_security_studies,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='sociology',
            hf_repo='lighteval/mmlu',
            hf_subset='sociology',
            evaluation_splits=['test'],
            prompt_function=mmlu_sociology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='us_foreign_policy',
            hf_repo='lighteval/mmlu',
            hf_subset='us_foreign_policy',
            evaluation_splits=['test'],
            prompt_function=mmlu_us_foreign_policy,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='virology',
            hf_repo='lighteval/mmlu',
            hf_subset='virology',
            evaluation_splits=['test'],
            prompt_function=mmlu_virology,
        ),
        create_evaluation_task(
            namespace='mmlu_reasoning',
            category='world_religions',
            hf_repo='lighteval/mmlu',
            hf_subset='world_religions',
            evaluation_splits=['test'],
            prompt_function=mmlu_world_religions,
        )
    ]


def create_ifeval_table():
    return [
        LightevalTaskConfig(
            name="ifeval_reasoning",
            prompt_function=lighteval.tasks.extended.ifeval.ifeval_prompt,
            suite=["community"],
            hf_repo="google/IFEval",
            hf_subset="default",
            metric=[lighteval.tasks.extended.ifeval.ifeval_metrics],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split="train",
            few_shots_select="random_sampling",
            generation_size=None,
            stop_sequence=[],
            version="0.1",
        )
    ]


def musr_choices(line, task_name: str = None):
    options = ast.literal_eval(line['choices'])
    letter_list = [c for c in LETTER_INDICES[:len(options)]]

    instruction = ''
    question = line['narrative'] + "\n\n"
    question += line['question'] + "\n\n"

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=options)

    return Doc(task_name=task_name, query=query, choices=letter_list, gold_index=line["answer_index"])


def create_musr_table():
    return [
        create_evaluation_task(
            namespace='musr_reasoning',
            category='murder_mysteries',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['murder_mysteries'],
            prompt_function=musr_choices
        ),
        create_evaluation_task(
            namespace='musr_reasoning',
            category='object_placements',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['object_placements'],
            prompt_function=musr_choices
        ),
        create_evaluation_task(
            namespace='musr_reasoning',
            category='team_allocation',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['team_allocation'],
            prompt_function=musr_choices
        )
    ]


def arc_with_options_letters_predict_custom_prompt(line, task_name: str = None):
    question = line['question']
    options = line["choices"]["text"]

    query = create_question_with_known_choices(instruction='', question=question, choices=options)

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["choices"]["text"])],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def create_arc_table():
    return [
        create_evaluation_task(
            namespace='arc_reasoning',
            category='easy',
            hf_repo='allenai/ai2_arc',
            hf_subset='ARC-Easy',
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=prompt.arc_with_options_letters_predict,
        ),
        create_evaluation_task(
            namespace='arc_reasoning',
            category='challenge',
            hf_repo='allenai/ai2_arc',
            hf_subset='ARC-Challenge',
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=prompt.arc_with_options_letters_predict,
        ),
        create_evaluation_task(
            namespace='arc_reasoning_custom_prompt',
            category='easy',
            hf_repo='allenai/ai2_arc',
            hf_subset='ARC-Easy',
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=arc_with_options_letters_predict_custom_prompt,
        ),
        create_evaluation_task(
            namespace='arc_reasoning_custom_prompt',
            category='challenge',
            hf_repo='allenai/ai2_arc',
            hf_subset='ARC-Challenge',
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=arc_with_options_letters_predict_custom_prompt,
        )
    ]


def truthful_qa_multiple_choice_mc1(line, task_name: str = None):
    instruction = ''
    question = {line['question']}
    
    choices = [f" {c}" for c in line["mc1_targets"]["choices"]]
    query = create_question_with_known_choices(instruction=instruction, question=question, choices=choices)

    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=[
            ix for ix, label in enumerate(line["mc1_targets"]["labels"]) if label == 1
        ],
        specific={"len_mc1": len(line["mc1_targets"]["choices"])},
    )


def truthful_qa_multiple_choice_mc2(line, task_name: str = None):
    instruction = ''
    question = {line['question']}
    
    choices = [f" {c}" for c in line["mc2_targets"]["choices"]]
    query = create_question_with_known_choices(instruction=instruction, question=question, choices=choices)

    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=[
            ix for ix, label in enumerate(line["mc2_targets"]["labels"]) if label == 1
        ],
        specific={"len_mc2": len(line["mc2_targets"]["choices"])},
    )


def create_truthfulqa_table():
    return [
        create_evaluation_task(
            namespace='truthfulqa_reasoning',
            category='mc1',
            hf_repo='truthful_qa',
            hf_subset='multiple_choice',
            evaluation_splits=['validation'],
            hf_avail_splits=['validation'],
            prompt_function=truthful_qa_multiple_choice_mc1,
        ),
        create_evaluation_task(
            namespace='truthfulqa_reasoning',
            category='mc2',
            hf_repo='truthful_qa',
            hf_subset='multiple_choice',
            evaluation_splits=['validation'],
            hf_avail_splits=['validation'],
            prompt_function=truthful_qa_multiple_choice_mc2,
        )
    ]


def hellaswag_generative(line, task_name: str = None):
    instruction = 'The following are multiple choice questions (with answers) about common sense.\n\n'
    question = f'Question: {line["activity_label"]}: {line["ctx_a"]} {line["ctx_b"].capitalize()}\n'
    choices = line['endings']

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=choices)

    gold_ix = int(line['label']) if line['label'] != "" else -1  # -1 for test
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_ix,
        instruction=instruction,
    )


def create_hellaswag_table():
    return [
        create_evaluation_task(
            namespace='hellaswag_reasoning',
            category='main',
            hf_repo='hellaswag',
            hf_subset=None,
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=hellaswag_generative,
        )
    ]



def social_iqa_with_cot(line, task_name: str = None):
    instruction = f"Answer questions about social situations and emotional intelligence given the provided context.\n\nContext: {line['context']}\n\n"

    choices = [line["answerA"], line["answerB"], line["answerC"]]

    question = line['question']
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )
    
    # Convert label (1, 2, 3) to index (0, 1, 2)
    gold_index = int(line["label"]) - 1 if line["label"].isdigit() else 0
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_index,
        instruction=query,
    )


def create_social_iqa_table():
    return [
        create_evaluation_task(
            namespace='social_iqa_reasoning',
            category='main',
            hf_repo='allenai/social_i_qa',
            hf_subset=None,
            evaluation_splits=['validation'],
            hf_avail_splits=['train', 'validation'],
            prompt_function=social_iqa_with_cot,
        )
    ]


def mctest_with_cot(line, task_name: str = None):
    instruction = f"Answer reading comprehension questions based on the given story.\n\nStory: {line['story']}\n\n"
    
    question = line['question']
    
    choices = [
        line["answer_options"]["A"],
        line["answer_options"]["B"], 
        line["answer_options"]["C"],
        line["answer_options"]["D"]
    ]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )
    
    answer_letter = line["answer"]
    gold_index = LETTER_INDICES.index(answer_letter)
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_index,
        instruction=query,
    )


def create_mctest_table():
    return [
        create_evaluation_task(
            namespace='mctest_reasoning',
            category='main',
            hf_repo='sagnikrayc/mctest',
            hf_subset=None,
            evaluation_splits=['test'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=mctest_with_cot,
        )
    ]


def piqa_with_cot(line, task_name: str = None):
    instruction = "Choose the most appropriate solution for the given goal."
    
    question = line["goal"]
    choices = [line["sol1"], line["sol2"]]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )
    
    gold_index = int(line["label"])
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_index,
        instruction=query,
    )

def create_piqa_table():
    return [
        create_evaluation_task(
            namespace='piqa_reasoning',
            category='main',
            hf_repo='ybisk/piqa',
            hf_subset=None,
            evaluation_splits=['validation'],
            hf_avail_splits=['train', 'validation'],
            prompt_function=piqa_with_cot,
        )
    ]


def commonsense_qa_with_cot(line, task_name: str = None):
    instruction = "Answer the following commonsense reasoning question."
    
    question = line["question"]
    choices = line["choices"]["text"]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )

    answer_key = line["answerKey"]
    gold_index = line["choices"]["label"].index(answer_key)
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_index,
        instruction=query,
    )


def create_commonsense_qa_table():
    return [
        create_evaluation_task(
            namespace='commonsense_qa_reasoning',
            category='main',
            hf_repo='tau/commonsense_qa',
            hf_subset=None,
            evaluation_splits=['validation'],
            hf_avail_splits=['train', 'validation', 'test'],
            prompt_function=commonsense_qa_with_cot,
        )
    ]


def global_mmlu_with_cot(line, task_name: str = None):
    subject = line["subject"]
    instruction = f"The following are multiple choice questions (with answers) about {subject}."
    
    question = line["question"]
    choices = [str(line["option_a"]), str(line["option_b"]), str(line["option_c"]), str(line["option_d"])]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )
    
    # Convert answer letter to index
    answer_letter = line["answer"]
    gold_index = LETTER_INDICES.index(answer_letter)
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_index,
        instruction=query,
    )


def salad_bench_with_cot(line, task_name: str = None):
    instruction = "Choose the most appropriate and safe response to the following question."
    
    question = line["question"]
    choices = line["choices"]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices
    )
    
    gold_indices = line["answers"]
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=gold_indices,
        instruction=query,
        specific={"all_correct_answers": gold_indices, "categories": line.get("categories", [])}
    )


def create_salad_bench_table():
    return [
        create_evaluation_task(
            namespace='salad_bench_reasoning',
            hf_subset='mrq',
            category='mrq',
            hf_repo='walledai/SaladBench',
            evaluation_splits=['train'],
            hf_avail_splits=['train'],
            prompt_function=salad_bench_with_cot,
        )
    ]


TASKS_TABLE = [
    *create_mmlu_table(),
    *create_bbh_table(),
    *create_gpqa_table(),
    *create_ifeval_table(),
    *create_musr_table(),
    *create_arc_table(),
    *create_hellaswag_table(),
    *create_truthfulqa_table(),
    *create_social_iqa_table(),
    *create_mctest_table(),
    *create_piqa_table(),
    *create_commonsense_qa_table(),
    *create_salad_bench_table()
]
