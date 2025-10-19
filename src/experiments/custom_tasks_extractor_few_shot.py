# NOTE: This file is long because lighteval copies the file to a temporary directory
# before running, therefore relative imports cannot be used
# TODO: Move most of the code into a package

from typing import List
import json
import os

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

import ast
import random


metric_name = 'letter'

# Specify full path to json due due to above note
json_file_path = ''
if json_file_path == '':
    raise ValueError('Please specify the full path to the "cot_template.json" file in the json_file_path variable in src/experiments/custom_tasks_extractor_few_shot.py")

USE_COT = True

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def create_cot_with_manual_choices(examples_data, choices, use_cot=USE_COT):
    """
    Create template with support for including/excluding reasoning traces.
    This version correctly includes the {Instruction} placeholder.
    """
    processed_examples = []
    for example in examples_data:
        question_with_choices = f'Question: {example["question"]}'
        for i, choice in enumerate(choices):
            question_with_choices += f'\n{LETTER_INDICES[i]}) {choice}'
        
        if use_cot:
            example_text = f'{question_with_choices}\nReasoning: {example["reasoning"]}\nAnswer: {example["answer"]}'
        else:
            example_text = f'{question_with_choices}\nAnswer: {example["answer"]}'
        processed_examples.append(example_text)

    template = '{Instruction}\n\n' + '\n\n'.join(processed_examples)
    template += '\n\nQuestion: {Question}'
    
    return template

def create_cot_with_existing_choices(examples_data, use_cot=USE_COT, format_question: bool = True):
    processed_examples = []
    for example in examples_data:
        if use_cot:
            if format_question:
                example_text = f'Question: {example["question_with_choices"]}\nReasoning: {example["reasoning"]}\nAnswer: {example["answer"]}'
            else:
                example_text = f'{example["question_with_choices"]}\nReasoning: {example["reasoning"]}\nAnswer: {example["answer"]}'
        else:
            if format_question:
                example_text = f'Question: {example["question_with_choices"]}\nAnswer: {example["answer"]}'
            else:
                example_text = f'{example["question_with_choices"]}\nAnswer: {example["answer"]}'
        processed_examples.append(example_text)
    
    template = '{Instruction}\n\n' + '\n\n'.join(processed_examples)
    template += '\n\nQuestion: {Question}'
    return template


def load_cot_templates(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        templates_data = json.load(f)
    
    templates = {}
    
    for template_name, template_config in templates_data.items():
        template_type = template_config.get('type')
        examples = template_config.get('examples', [])
        
        if template_type == 'manual_choices':
            choices = template_config.get('choices', [])
            template = create_cot_with_manual_choices(examples, choices)
        elif template_type == 'existing_choices':
            template = create_cot_with_existing_choices(examples)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
        
        templates[template_name] = template
    
    return templates


cot_templates = load_cot_templates(json_file_path)

BBH_BOOLEAN_EXPRESSIONS_COT = cot_templates.get('bbh_boolean_expressions')
BBH_CAUSAL_JUDGEMENT_COT = cot_templates.get('bbh_causal_judgement')
BBH_SPORTS_UNDERSTANDING_COT = cot_templates.get('bbh_sports_understanding')
BBH_WEB_OF_LIES_COT = cot_templates.get('bbh_web_of_lies')
BBH_NAVIGATE_COT = cot_templates.get('bbh_navigate')
BBH_FORMAL_FALLACIES_COT = cot_templates.get('bbh_formal_fallacies')
BBH_OBJECT_COUNTING_COT = cot_templates.get('bbh_object_counting')
BBH_MULTISTEP_ARITHMETIC_TWO_COT = cot_templates.get('bbh_multistep_arithmetic_two')
BBH_MOVIE_RECOMMENDATION_COT = cot_templates.get('bbh_movie_recommendation')
BBH_DYCK_LANGUAGES_COT = cot_templates.get('bbh_dyck_languages')
BBH_WORD_SORTING_COT = cot_templates.get('bbh_word_sorting')
BBH_DATE_UNDERSTANDING_COT = cot_templates.get('bbh_date_understanding')
BBH_DISAMBIGUATION_QA_COT = cot_templates.get('bbh_disambiguation_qa')
BBH_SNARKS_COT = cot_templates.get('bbh_snarks')
BBH_GEOMETRIC_SHAPES_COT = cot_templates.get('bbh_geometric_shapes')
BBH_HYPERBATON_COT = cot_templates.get('bbh_hyperbaton')
BBH_LOGICAL_DEDUCTION_THREE_OBJECTS_COT = cot_templates.get('bbh_logical_deduction_three_objects')
BBH_LOGICAL_DEDUCTION_FIVE_OBJECTS_COT = cot_templates.get('bbh_logical_deduction_five_objects')
BBH_LOGICAL_DEDUCTION_SEVEN_OBJECTS_COT = cot_templates.get('bbh_logical_deduction_seven_objects')
BBH_TEMPORAL_SEQUENCES_COT = cot_templates.get('bbh_temporal_sequences')
BBH_TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS_COT = cot_templates.get('bbh_tracking_shuffled_objects_three_objects')
BBH_TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS_COT = cot_templates.get('bbh_tracking_shuffled_objects_five_objects')
BBH_TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS_COT = cot_templates.get('bbh_tracking_shuffled_objects_seven_objects')
BBH_PENGUINS_IN_A_TABLE_COT = cot_templates.get('bbh_penguins_in_a_table')
BBH_REASONING_ABOUT_COLORED_OBJECTS_COT = cot_templates.get('bbh_reasoning_about_colored_objects')
BBH_RUIN_NAMES_COT = cot_templates.get('bbh_ruin_names')
BBH_SALIENT_TRANSLATION_ERROR_DETECTION_COT = cot_templates.get('bbh_salient_translation_error_detection')
MUSR_MURDER_MYSTERIES_COT = cot_templates.get('musr_murder_mysteries')
MUSR_OBJECT_PLACEMENTS_COT = cot_templates.get('musr_object_placements')
MUSR_TEAM_ALLOCATION_COT = cot_templates.get('musr_team_allocation')
SOCIAL_IQA_COT = cot_templates.get('social_iqa')
ARC_COT = cot_templates.get('arc')
MCTEST_COT = cot_templates.get('mctest')
GPQA_COT = cot_templates.get('gpqa')
HELLASWAG_COT = cot_templates.get('hellaswag')
PIQA_COT = cot_templates.get('piqa')
COMMONSENSE_QA_COT = cot_templates.get('commonsense_qa')

mmlu_cot_templates = {}
with open(json_file_path, 'r', encoding='utf-8') as f:
    templates_data = json.load(f)
    for template_name, template_config in templates_data.items():
        if template_name.startswith('mmlu_'):
            template_type = template_config.get('type')
            examples = template_config.get('examples', [])
            if template_type == 'manual_choices':
                choices = template_config.get('choices', [])
                mmlu_cot_templates[template_name] = create_cot_with_manual_choices(examples, choices, use_cot=False)
            elif template_type == 'existing_choices':
                mmlu_cot_templates[template_name] = create_cot_with_existing_choices(examples, use_cot=False)


MMLU_ABSTRACT_ALGEBRA_COT = mmlu_cot_templates.get('mmlu_abstract_algebra')
MMLU_ANATOMY_COT = mmlu_cot_templates.get('mmlu_anatomy')
MMLU_ASTRONOMY_COT = mmlu_cot_templates.get('mmlu_astronomy')
MMLU_BUSINESS_ETHICS_COT = mmlu_cot_templates.get('mmlu_business_ethics')
MMLU_CLINICAL_KNOWLEDGE_COT = mmlu_cot_templates.get('mmlu_clinical_knowledge')
MMLU_COLLEGE_BIOLOGY_COT = mmlu_cot_templates.get('mmlu_college_biology')
MMLU_COLLEGE_CHEMISTRY_COT = mmlu_cot_templates.get('mmlu_college_chemistry')
MMLU_COLLEGE_COMPUTER_SCIENCE_COT = mmlu_cot_templates.get('mmlu_college_computer_science')
MMLU_COLLEGE_MATHEMATICS_COT = mmlu_cot_templates.get('mmlu_college_mathematics')
MMLU_COLLEGE_MEDICINE_COT = mmlu_cot_templates.get('mmlu_college_medicine')
MMLU_COLLEGE_PHYSICS_COT = mmlu_cot_templates.get('mmlu_college_physics')
MMLU_COMPUTER_SECURITY_COT = mmlu_cot_templates.get('mmlu_computer_security')
MMLU_CONCEPTUAL_PHYSICS_COT = mmlu_cot_templates.get('mmlu_conceptual_physics')
MMLU_ECONOMETRICS_COT = mmlu_cot_templates.get('mmlu_econometrics')
MMLU_ELECTRICAL_ENGINEERING_COT = mmlu_cot_templates.get('mmlu_electrical_engineering')
MMLU_ELEMENTARY_MATHEMATICS_COT = mmlu_cot_templates.get('mmlu_elementary_mathematics')
MMLU_FORMAL_LOGIC_COT = mmlu_cot_templates.get('mmlu_formal_logic')
MMLU_GLOBAL_FACTS_COT = mmlu_cot_templates.get('mmlu_global_facts')
MMLU_HIGH_SCHOOL_BIOLOGY_COT = mmlu_cot_templates.get('mmlu_high_school_biology')
MMLU_HIGH_SCHOOL_CHEMISTRY_COT = mmlu_cot_templates.get('mmlu_high_school_chemistry')
MMLU_HIGH_SCHOOL_COMPUTER_SCIENCE_COT = mmlu_cot_templates.get('mmlu_high_school_computer_science')
MMLU_HIGH_SCHOOL_EUROPEAN_HISTORY_COT = mmlu_cot_templates.get('mmlu_high_school_european_history')
MMLU_HIGH_SCHOOL_GEOGRAPHY_COT = mmlu_cot_templates.get('mmlu_high_school_geography')
MMLU_HIGH_SCHOOL_GOVERNMENT_AND_POLITICS_COT = mmlu_cot_templates.get('mmlu_high_school_government_and_politics')
MMLU_HIGH_SCHOOL_MACROECONOMICS_COT = mmlu_cot_templates.get('mmlu_high_school_macroeconomics')
MMLU_HIGH_SCHOOL_MATHEMATICS_COT = mmlu_cot_templates.get('mmlu_high_school_mathematics')
MMLU_HIGH_SCHOOL_MICROECONOMICS_COT = mmlu_cot_templates.get('mmlu_high_school_microeconomics')
MMLU_HIGH_SCHOOL_PHYSICS_COT = mmlu_cot_templates.get('mmlu_high_school_physics')
MMLU_HIGH_SCHOOL_PSYCHOLOGY_COT = mmlu_cot_templates.get('mmlu_high_school_psychology')
MMLU_HIGH_SCHOOL_STATISTICS_COT = mmlu_cot_templates.get('mmlu_high_school_statistics')
MMLU_HIGH_SCHOOL_US_HISTORY_COT = mmlu_cot_templates.get('mmlu_high_school_us_history')
MMLU_HIGH_SCHOOL_WORLD_HISTORY_COT = mmlu_cot_templates.get('mmlu_high_school_world_history')
MMLU_HUMAN_AGING_COT = mmlu_cot_templates.get('mmlu_human_aging')
MMLU_HUMAN_SEXUALITY_COT = mmlu_cot_templates.get('mmlu_human_sexuality')
MMLU_INTERNATIONAL_LAW_COT = mmlu_cot_templates.get('mmlu_international_law')
MMLU_JURISPRUDENCE_COT = mmlu_cot_templates.get('mmlu_jurisprudence')
MMLU_LOGICAL_FALLACIES_COT = mmlu_cot_templates.get('mmlu_logical_fallacies')
MMLU_MACHINE_LEARNING_COT = mmlu_cot_templates.get('mmlu_machine_learning')
MMLU_MANAGEMENT_COT = mmlu_cot_templates.get('mmlu_management')
MMLU_MARKETING_COT = mmlu_cot_templates.get('mmlu_marketing')
MMLU_MEDICAL_GENETICS_COT = mmlu_cot_templates.get('mmlu_medical_genetics')
MMLU_MISCELLANEOUS_COT = mmlu_cot_templates.get('mmlu_miscellaneous')
MMLU_MORAL_DISPUTES_COT = mmlu_cot_templates.get('mmlu_moral_disputes')
MMLU_MORAL_SCENARIOS_COT = mmlu_cot_templates.get('mmlu_moral_scenarios')
MMLU_NUTRITION_COT = mmlu_cot_templates.get('mmlu_nutrition')
MMLU_PHILOSOPHY_COT = mmlu_cot_templates.get('mmlu_philosophy')
MMLU_PREHISTORY_COT = mmlu_cot_templates.get('mmlu_prehistory')
MMLU_PROFESSIONAL_ACCOUNTING_COT = mmlu_cot_templates.get('mmlu_professional_accounting')
MMLU_PROFESSIONAL_LAW_COT = mmlu_cot_templates.get('mmlu_professional_law')
MMLU_PROFESSIONAL_MEDICINE_COT = mmlu_cot_templates.get('mmlu_professional_medicine')
MMLU_PROFESSIONAL_PSYCHOLOGY_COT = mmlu_cot_templates.get('mmlu_professional_psychology')
MMLU_PUBLIC_RELATIONS_COT = mmlu_cot_templates.get('mmlu_public_relations')
MMLU_SECURITY_STUDIES_COT = mmlu_cot_templates.get('mmlu_security_studies')
MMLU_SOCIOLOGY_COT = mmlu_cot_templates.get('mmlu_sociology')
MMLU_US_FOREIGN_POLICY_COT = mmlu_cot_templates.get('mmlu_us_foreign_policy')
MMLU_VIROLOGY_COT = mmlu_cot_templates.get('mmlu_virology')
MMLU_WORLD_RELIGIONS_COT = mmlu_cot_templates.get('mmlu_world_religions')


COT_QUESTION = """
Question: {Question}
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
    stop_sequence=None,
    # use_cot: bool = USE_COT,
    include_topic: bool = False
) -> LightevalTaskConfig:
    
    if stop_sequence is None:
        stop_sequence = ['<|im_start|>', 'Question: ', '\nQuestion:', 'Question: ', '\nQuestion: ', 'You are a helpful assistant', 'You are an AI assistant']

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


def create_question_without_known_choices(instruction: str, question: str, cot_template: str = None, use_cot: bool = USE_COT, include_topic: bool = False, topic: str = ""):
    if include_topic and topic:
        if instruction:
            instruction = f"The following are multiple choice questions (with answers) about {topic}.\n\n{instruction}"
        else:
            instruction = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    
    if cot_template:
        query = [cot_template.format(Instruction=instruction, Question=question)]
    else:
        query = [COT_QUESTION.format(Instruction=instruction, Question=question)]

    if use_cot:
        query.append("Reasoning:")
    else:
        query.append("Answer:")
    
    return '\n'.join(query).strip()


def create_question_with_known_choices(instruction: str, question: str, choices: List[str], cot_template: str = None, use_cot: bool = USE_COT, include_topic: bool = False, topic: str = ""):
    if include_topic and topic:
        if instruction:
            instruction = f"The following are multiple choice questions (with answers) about {topic}.\n\n{instruction}"
        else:
            instruction = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    
    if cot_template:
        query = [cot_template.format(Instruction=instruction, Question=question)]
    else:
        query = [COT_QUESTION.format(Instruction=instruction, Question=question)]

    for i, choice in enumerate(choices):
        query.append(f'{LETTER_INDICES[i]}) {choice}')
    
    if use_cot:
        query.append("Reasoning:")
    else:
        query.append("Answer:")
    
    return '\n'.join(query).strip()

def bbh_with_cot_with_known_choices(line, task_name: str, instruction: str, choices: List[str], cot_template: str = None, use_cot: bool = USE_COT, include_topic: bool = False, topic: str = ""):
    gold_tf = line["target"]
    gold_index_tf = choices.index(gold_tf)
    gold_letter = LETTER_INDICES[:len(choices)][gold_index_tf]
    
    query = create_question_with_known_choices(
        instruction=instruction, 
        question=line['input'], 
        choices=choices, 
        cot_template=cot_template,
        use_cot=use_cot,
        include_topic=include_topic,
        topic=topic
    )

    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=letter_list.index(gold_letter),
        instruction=query,
    )

def bbh_with_cot_without_known_choices(line, task_name: str, instruction: str, n_choices: int, use_numeric: bool = False, cot_template: str = None, use_cot: bool = USE_COT, include_topic: bool = False, topic: str = ""):
    choices = [f"({c})" for c in LETTER_INDICES[:n_choices]]
    if use_numeric:
        choices = [f"{c}" for c in range(1, n_choices + 1)]

    gold_tf = line["target"]
    gold_index_tf = choices.index(gold_tf)
    gold_letter = LETTER_INDICES[:n_choices][gold_index_tf]
    
    query = create_question_without_known_choices(
        instruction=instruction, 
        question=line['input'], 
        cot_template=cot_template,
        use_cot=use_cot,
        include_topic=include_topic,
        topic=topic
    )

    letter_list = [c for c in LETTER_INDICES[:n_choices]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=letter_list,
        gold_index=letter_list.index(gold_letter),
        instruction=query,
    )


def bbh_boolean_expressions(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Evaluate the result of a random Boolean expression."
    topic = "boolean expressions" if include_topic else ""
    choices = ["False", "True"]
    template = BBH_BOOLEAN_EXPRESSIONS_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_causal_judgement(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Answer questions about causal attribution."
    topic = "causal attribution" if include_topic else ""
    choices = ["Yes", "No"]
    template = BBH_CAUSAL_JUDGEMENT_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_date_understanding(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Infer the date from context."
    topic = "date understanding" if include_topic else ""
    template = BBH_DATE_UNDERSTANDING_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_disambiguation_qa(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Clarify the meaning of sentences with ambiguous pronouns."
    topic = "pronoun disambiguation" if include_topic else ""
    template = BBH_DISAMBIGUATION_QA_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_dyck_languages(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Correctly close a Dyck-n word."
    topic = "formal languages" if include_topic else ""
    choices = [line["target"]]
    template = BBH_DYCK_LANGUAGES_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_formal_fallacies(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Distinguish deductively valid arguments from formal fallacies."
    topic = "logical reasoning" if include_topic else ""
    choices = ["valid", "invalid"]
    template = BBH_FORMAL_FALLACIES_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_geometric_shapes(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Name geometric shapes from their SVG paths."
    topic = "geometry" if include_topic else ""
    template = BBH_GEOMETRIC_SHAPES_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=11, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_hyperbaton(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Order adjectives correctly in English sentences."
    topic = "English grammar" if include_topic else ""
    template = BBH_HYPERBATON_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=2, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_logical_deduction_five_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects."
    topic = "logical reasoning" if include_topic else ""
    template = BBH_LOGICAL_DEDUCTION_FIVE_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_logical_deduction_seven_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects."
    topic = "logical reasoning" if include_topic else ""
    template = BBH_LOGICAL_DEDUCTION_SEVEN_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=7, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_logical_deduction_three_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"
    topic = "logical reasoning" if include_topic else ""
    template = BBH_LOGICAL_DEDUCTION_THREE_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_movie_recommendation(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    if line["target"] == "Monsters, Inc":  # this line is not correctly formatted
        print(
            "One sample removed from task bbh:movie_recommendation because its line is incorrectly formatted."
        )
        return []
    instruction = "Recommend movies similar to the given list of movies."
    topic = "movie recommendations" if include_topic else ""
    template = BBH_MOVIE_RECOMMENDATION_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_multistep_arithmetic_two(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Solve multi-step arithmetic problems."
    topic = "arithmetic" if include_topic else ""
    choices = [line["target"]]
    template = BBH_MULTISTEP_ARITHMETIC_TWO_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_navigate(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Given a series of navigation instructions, determine whether one would end up back at the starting point."
    topic = "spatial reasoning" if include_topic else ""
    choices = ["Yes", "No"]
    template = BBH_NAVIGATE_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_object_counting(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Questions that involve enumerating objects and asking the model to count them."
    topic = "counting" if include_topic else ""
    template = BBH_OBJECT_COUNTING_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=19, use_numeric=True, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_penguins_in_a_table(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Answer questions about a table of penguins and their attributes."
    topic = "table reasoning" if include_topic else ""
    template = BBH_PENGUINS_IN_A_TABLE_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_reasoning_about_colored_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Answer extremely simple questions about the colors of objects on a surface."
    topic = "visual reasoning" if include_topic else ""
    template = BBH_REASONING_ABOUT_COLORED_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=18, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_ruin_names(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    if line["target"] in ["dearth, wind, & fire", "rita, sue and bob poo"]:  # line not correctly formatted
        print("One sample removed from task bbh:ruin_names because its line is incorrectly formatted.")
        return []
    instruction = "Select the humorous edit that 'ruins' the input movie or musical artist name."
    topic = "humor" if include_topic else ""
    template = BBH_RUIN_NAMES_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_salient_translation_error_detection(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Detect the type of error in an English translation of a German source sentence."
    topic = "translation" if include_topic else ""
    template = BBH_SALIENT_TRANSLATION_ERROR_DETECTION_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=6, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_snarks(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = 'Determine which of two sentences is sarcastic.\n\nAccording to Cambridge University Dictionary, sarcasm is "the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone\'s feelings or to criticize something in a humorous way." Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.'
    topic = "sarcasm detection" if include_topic else ""
    template = BBH_SNARKS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=2, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_sports_understanding(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Determine whether an artificially constructed sentence relating to sports is plausible or not."
    topic = "sports understanding" if include_topic else ""
    choices = ["yes", "no"]
    template = BBH_SPORTS_UNDERSTANDING_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_temporal_sequences(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Task description: Answer questions about which times certain events could have occurred.\n\n"
    topic = "temporal reasoning" if include_topic else ""
    template = BBH_TEMPORAL_SEQUENCES_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=4, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_tracking_shuffled_objects_five_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps."
    topic = "object tracking" if include_topic else ""
    template = BBH_TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=5, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_tracking_shuffled_objects_seven_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps."
    topic = "object tracking" if include_topic else ""
    template = BBH_TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=7, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_tracking_shuffled_objects_three_objects(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps."
    topic = "object tracking" if include_topic else ""
    template = BBH_LOGICAL_DEDUCTION_THREE_OBJECTS_COT
    return bbh_with_cot_without_known_choices(line, task_name, instruction, n_choices=3, cot_template=template, use_cot=use_cot, include_topic=include_topic, topic=topic)


def bbh_web_of_lies(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Evaluate a random boolean function expressed as a word problem."
    topic = "logical reasoning" if include_topic else ""
    choices = ["Yes", "No"]
    template = BBH_WEB_OF_LIES_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


def bbh_word_sorting(line, task_name: str = None, use_cot: bool = USE_COT, include_topic: bool = False):
    instruction = "Sort a list of words."
    topic = "alphabetical sorting" if include_topic else ""
    choices = [line["target"]]
    template = BBH_WORD_SORTING_COT
    return bbh_with_cot_with_known_choices(line, task_name, instruction, choices, template, use_cot, include_topic, topic)


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
        choices=choices,
        cot_template=GPQA_COT
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


def musr_murder_mysteries_choices(line, task_name: str = None):
    options = ast.literal_eval(line['choices'])
    letter_list = [c for c in LETTER_INDICES[:len(options)]]

    instruction = ''
    question = line['narrative'] + "\n\n"
    question += line['question']

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=options, cot_template=MUSR_MURDER_MYSTERIES_COT)

    return Doc(task_name=task_name, query=query, choices=letter_list, gold_index=line["answer_index"])


def musr_object_placements_choices(line, task_name: str = None):
    options = ast.literal_eval(line['choices'])
    letter_list = [c for c in LETTER_INDICES[:len(options)]]

    instruction = ''
    question = line['narrative'] + "\n\n"
    question += line['question']

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=options, cot_template=MUSR_OBJECT_PLACEMENTS_COT)

    return Doc(task_name=task_name, query=query, choices=letter_list, gold_index=line["answer_index"])


def musr_team_allocation_choices(line, task_name: str = None):
    options = ast.literal_eval(line['choices'])
    letter_list = [c for c in LETTER_INDICES[:len(options)]]

    instruction = ''
    question = line['narrative'] + "\n\n"
    question += line['question']

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=options, cot_template=MUSR_TEAM_ALLOCATION_COT)

    return Doc(task_name=task_name, query=query, choices=letter_list, gold_index=line["answer_index"])


def create_musr_table():
    return [
        create_evaluation_task(
            namespace='musr_reasoning',
            category='murder_mysteries',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['murder_mysteries'],
            prompt_function=musr_murder_mysteries_choices
        ),
        create_evaluation_task(
            namespace='musr_reasoning',
            category='object_placements',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['object_placements'],
            prompt_function=musr_object_placements_choices
        ),
        create_evaluation_task(
            namespace='musr_reasoning',
            category='team_allocation',
            hf_repo='TAUR-Lab/MuSR',
            hf_subset=None,
            evaluation_splits=['team_allocation'],
            prompt_function=musr_team_allocation_choices
        )
    ]


def arc_with_options_letters_predict_custom_prompt(line, task_name: str = None):
    question = line['question']
    options = line["choices"]["text"]

    query = create_question_with_known_choices(instruction='', question=question, choices=options, cot_template=ARC_COT)

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


def hellaswag_generative(line, task_name: str = None):
    instruction = 'The following are multiple choice questions (with answers) about common sense.'
    question = f'Question: {line["activity_label"]}: {line["ctx_a"]} {line["ctx_b"].capitalize()}\n'
    choices = line['endings']

    query = create_question_with_known_choices(instruction=instruction, question=question, choices=choices, cot_template=HELLASWAG_COT)

    gold_ix = int(line['label']) if line['label'] != "" else -1
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
    instruction = "Answer questions about social situations and emotional intelligence given the provided context."
    
    question = f"{line['context']}\n\n{line['question']}"
    choices = [line["answerA"], line["answerB"], line["answerC"]]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices,
        cot_template=SOCIAL_IQA_COT
    )
    
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
    instruction = "Answer reading comprehension questions based on the given story."
    question = f"{line['story']}\n\n{line['question']}"    

    choices = [
        line["answer_options"]["A"],
        line["answer_options"]["B"], 
        line["answer_options"]["C"],
        line["answer_options"]["D"]
    ]
    
    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices,
        cot_template=MCTEST_COT
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
        choices=choices,
        cot_template=PIQA_COT,
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
        choices=choices,
        cot_template=COMMONSENSE_QA_COT,
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


def create_mmlu_few_shot_template(examples_data, use_cot=USE_COT, topic=""):
    if use_cot:
        raise ValueError("Chain-of-thought prompting is not yet supported for MMLU few-shot. Use use_cot=False for standard few-shot prompting.")
    
    processed_examples = []
    for example in examples_data:
        question_text = f'Question: {example["question"]}'
        for i, choice in enumerate(example["choices"]):
            question_text += f'\n{LETTER_INDICES[i]}) {choice}'
        
        answer_letter = LETTER_INDICES[example["answer"]]
        example_text = f'{question_text}\nAnswer: {answer_letter}'
        processed_examples.append(example_text)
    
    instruction = ""
    if topic:
        instruction = f"The following are multiple choice questions (with answers) about {topic}."
    
    template = instruction + '\n\n'.join(processed_examples) + '\n\nQuestion: {Question}'
    return template


def mmlu_with_cot(line: dict, task_name: str, few_shot_template: str, topic: str, include_topic: bool = True):
    if not few_shot_template:
        raise ValueError(f"No few-shot template provided for MMLU task: {task_name}.")

    question = line["question"]
    choices = line["choices"]

    instruction = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}." if include_topic else ""

    query = create_question_with_known_choices(
        instruction=instruction,
        question=question,
        choices=choices,
        cot_template=few_shot_template
    )

    gold_ix = line["answer"] if isinstance(line["answer"], int) else LETTER_INDICES.index(line["answer"])
    letter_list = [c for c in LETTER_INDICES[:len(choices)]]

    return Doc(
        task_name=task_name,
        query=query.strip(),
        choices=letter_list,
        gold_index=gold_ix,
        instruction=query.strip(),
    )



def mmlu_abstract_algebra(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ABSTRACT_ALGEBRA_COT, "abstract_algebra")

def mmlu_anatomy(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ANATOMY_COT, "anatomy")

def mmlu_astronomy(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ASTRONOMY_COT, "astronomy")

def mmlu_business_ethics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_BUSINESS_ETHICS_COT, "business_ethics")

def mmlu_clinical_knowledge(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_CLINICAL_KNOWLEDGE_COT, "clinical_knowledge")

def mmlu_college_biology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_BIOLOGY_COT, "college_biology")

def mmlu_college_chemistry(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_CHEMISTRY_COT, "college_chemistry")

def mmlu_college_computer_science(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_COMPUTER_SCIENCE_COT, "college_computer_science")

def mmlu_college_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_MATHEMATICS_COT, "college_mathematics")

def mmlu_college_medicine(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_MEDICINE_COT, "college_medicine")

def mmlu_college_physics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COLLEGE_PHYSICS_COT, "college_physics")

def mmlu_computer_security(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_COMPUTER_SECURITY_COT, "computer_security")

def mmlu_conceptual_physics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_CONCEPTUAL_PHYSICS_COT, "conceptual_physics")

def mmlu_econometrics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ECONOMETRICS_COT, "econometrics")

def mmlu_electrical_engineering(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ELECTRICAL_ENGINEERING_COT, "electrical_engineering")

def mmlu_elementary_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_ELEMENTARY_MATHEMATICS_COT, "elementary_mathematics")

def mmlu_formal_logic(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_FORMAL_LOGIC_COT, "formal_logic")

def mmlu_global_facts(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_GLOBAL_FACTS_COT, "global_facts")

def mmlu_high_school_biology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_BIOLOGY_COT, "high_school_biology")

def mmlu_high_school_chemistry(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_CHEMISTRY_COT, "high_school_chemistry")

def mmlu_high_school_computer_science(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_COMPUTER_SCIENCE_COT, "high_school_computer_science")

def mmlu_high_school_european_history(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_EUROPEAN_HISTORY_COT, "high_school_european_history")

def mmlu_high_school_geography(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_GEOGRAPHY_COT, "high_school_geography")

def mmlu_high_school_government_and_politics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_GOVERNMENT_AND_POLITICS_COT, "high_school_government_and_politics")

def mmlu_high_school_macroeconomics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_MACROECONOMICS_COT, "high_school_macroeconomics")

def mmlu_high_school_mathematics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_MATHEMATICS_COT, "high_school_mathematics")

def mmlu_high_school_microeconomics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_MICROECONOMICS_COT, "high_school_microeconomics")

def mmlu_high_school_physics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_PHYSICS_COT, "high_school_physics")

def mmlu_high_school_psychology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_PSYCHOLOGY_COT, "high_school_psychology")

def mmlu_high_school_statistics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_STATISTICS_COT, "high_school_statistics")

def mmlu_high_school_us_history(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_US_HISTORY_COT, "high_school_us_history")

def mmlu_high_school_world_history(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HIGH_SCHOOL_WORLD_HISTORY_COT, "high_school_world_history")

def mmlu_human_aging(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HUMAN_AGING_COT, "human_aging")

def mmlu_human_sexuality(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_HUMAN_SEXUALITY_COT, "human_sexuality")

def mmlu_international_law(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_INTERNATIONAL_LAW_COT, "international_law")

def mmlu_jurisprudence(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_JURISPRUDENCE_COT, "jurisprudence")

def mmlu_logical_fallacies(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_LOGICAL_FALLACIES_COT, "logical_fallacies")

def mmlu_machine_learning(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MACHINE_LEARNING_COT, "machine_learning")

def mmlu_management(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MANAGEMENT_COT, "management")

def mmlu_marketing(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MARKETING_COT, "marketing")

def mmlu_medical_genetics(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MEDICAL_GENETICS_COT, "medical_genetics")

def mmlu_miscellaneous(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MISCELLANEOUS_COT, "miscellaneous")

def mmlu_moral_disputes(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MORAL_DISPUTES_COT, "moral_disputes")

def mmlu_moral_scenarios(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_MORAL_SCENARIOS_COT, "moral_scenarios")

def mmlu_nutrition(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_NUTRITION_COT, "nutrition")

def mmlu_philosophy(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PHILOSOPHY_COT, "philosophy")

def mmlu_prehistory(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PREHISTORY_COT, "prehistory")

def mmlu_professional_accounting(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PROFESSIONAL_ACCOUNTING_COT, "professional_accounting")

def mmlu_professional_law(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PROFESSIONAL_LAW_COT, "professional_law")

def mmlu_professional_medicine(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PROFESSIONAL_MEDICINE_COT, "professional_medicine")

def mmlu_professional_psychology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PROFESSIONAL_PSYCHOLOGY_COT, "professional_psychology")

def mmlu_public_relations(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_PUBLIC_RELATIONS_COT, "public_relations")

def mmlu_security_studies(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_SECURITY_STUDIES_COT, "security_studies")

def mmlu_sociology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_SOCIOLOGY_COT, "sociology")

def mmlu_us_foreign_policy(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_US_FOREIGN_POLICY_COT, "us_foreign_policy")

def mmlu_virology(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_VIROLOGY_COT, "virology")

def mmlu_world_religions(line, task_name: str = None):
    return mmlu_with_cot(line, task_name, MMLU_WORLD_RELIGIONS_COT, "world_religions")

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


TASKS_TABLE = [
    *create_bbh_table(),
    *create_gpqa_table(),
    *create_musr_table(),
    *create_arc_table(),
    *create_hellaswag_table(),
    *create_social_iqa_table(),
    *create_mctest_table(),
    *create_piqa_table(),
    *create_commonsense_qa_table(),
    *create_mmlu_table(),
]
