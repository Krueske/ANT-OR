"""
Prompts for ANT-OR Framework

This module contains the prompts used by the annotation framework.
Only prompts actually used in the codebase are included.
"""

# English prompts (used by the framework)
PROMPTS_EN = {
    # ============================================================================
    # Basic Annotation Prompts
    # ============================================================================

    "annotation_based_on_generated_prompt": """{generated_prompt}

### Query to be Annotated:
{query}

### Please output in the following format:
Reasoning Process: [Step-by-step reasoning]
Annotation Result: {output_format}
""",

    "annotation_based_on_generated_prompt_direct": """{generated_prompt}

### Query to be Annotated:
{query}

### Please output in the following format:
Annotation Result: {output_format}
""",

    "icl_annotation_prompt": """You are a professional annotation expert. Please annotate the following Query according to the rules and examples below.

### Annotation Rules:
{doc}

### Reference Examples:
{icl_context}

### Query to Annotate:
{query}

### Please output in the following format:
Reasoning process: [Step-by-step analysis]
Annotation result: {output_format}
""",

    "icl_example_section": """[Example {idx}] (similarity: {similarity:.3f})
Query: {query_with_images}
Ground Truth: {ground_truth}
""",

    # ============================================================================
    # Verification Prompts
    # ============================================================================

    "default_verification_prompt_template": """Check the reasoning logic of multiple annotators and select the best result.

Focus on:
1. Compare the reasoning logic of each annotator
2. Check for logical consistency and coherence
3. Identify any reasoning errors or inconsistencies
4. Select the annotator with the most sound reasoning

Note: When all three annotators disagree (three-way tie), you must select the best one based on reasoning quality.""",

    "verification_with_logic_scoring": """Check the reasoning logic of multiple annotators and select the best result.

Task:
1. Check each annotator's reasoning for logical errors (contradictions, gaps, circular reasoning)
2. Score each annotator's logic quality (0-10)
3. Select the result with the best reasoning

Note: When all three annotators disagree (three-way tie), you must select the best one based on reasoning quality.""",

    "verification_prompt_format": """{verification_prompt_template}

Query: {query_str}{dimension_section}

Annotations:{annotations_text}
Vote Info: {vote_text}{sample_section}

Output format:
```json
{{
  "reason": ["logic error description for each annotator"],
  "result": ["annotator1/annotator2/annotator3"]
}}
```""",

    # ============================================================================
    # Dimension Mining & Analysis Prompts
    # ============================================================================

    "analyze_contrastive_triple": """Analyze this group of cases that are 'similar but judged differently' to deeply uncover the **fundamental intent** behind the rule formulation.

These samples are very similar in semantic input, but some are correctly judged while others are incorrectly judged.

### Original Rule Document:
{doc_content}

### Historical Discriminative Dimensions:
Before inducing new dimensions, please check the following existing discriminative dimensions. If you find that the discriminative dimension reflected in the current case is **substantially the same** as a historical dimension (i.e., describing the same type of meta-rule), please directly use that historical dimension's meta-rule as the name instead of creating a new name.

{existing_dims_text}

### Case group to analyze:
{case_text}

Please deeply analyze this contrastive case group to uncover the **fundamental intent** behind rule formulation:

1. [Surface Differences] What are the literal differences between samples? Especially between correct and incorrect samples
2. [Decision Boundary] What factors cause samples to be judged as correct or incorrect? (How should similar samples be annotated?)
3. [Rule Intent] What is the **core reason** why rule makers require this annotation? (i.e., what is the general core annotation logic?)
4. [Meta-Rule] If this distinguishing principle is abstracted into a general rule, how should it be expressed?
5. [Applicable Boundary] Under what circumstances does this rule apply? When might it not apply?

Please note: Please analyze carefully based on the samples themselves. Since some samples may have deviations, please avoid using overly absolute judgment logic.

Please output your complete reasoning process first, then output the final JSON result:

Reasoning process: [Your reasoning process]

Output JSON format:
```json
{{
  "surface_difference": "Description of surface differences",
  "decision_boundary": "Decision boundary (How should similar samples be annotated?)",
  "rule_intent": "Core logic of rule formulation (i.e., general task annotation logic)",
  "meta_rule": "Reusable meta-rule expression (used as dimension name, prioritize using historical dimension names)",
  "applicable_boundary": "Applicable boundary conditions (what types of samples it applies to)",
}}
```

Note:
- meta_rule will be used as the unique identifier name for the discriminative dimension; if it is substantially the same as a historical dimension, please use the historical dimension name
""",

    "refine_dimension_fields": """
You are an annotation rule expert. Based on the main rule document, integrate historical rules with new cases to extract **generalizable** rule intent (core annotation logic).

### Definitions

1. [Rule Intent] What is the **core reason** why rule makers require this annotation? (i.e., the general core annotation logic)
2. [Decision Boundary] What factors cause one case to be judged as X and another as Y?
3. [Applicable Boundary] Under what circumstances does this rule apply? When might it not apply?

### Input Information

[Original Rule Document]
{doc_content}

[Historical Discriminative Dimensions]
- Rule Intent: {existing_intent}
- Decision Boundary: {existing_boundary}
- Applicable Boundary: {existing_applicable}

[New Case Analysis]
{case_text}
- Preliminary intent: {new_intent}
- Preliminary boundary: {new_boundary}
- Preliminary applicability: {new_applicable}

### Fusion Analysis Steps (Please follow this reasoning)

1. **Relationship Determination**: Determine the relationship between the new case and historical rules
   - Support: New case validates and reinforces historical rules
   - Supplement: Fills gaps in scenarios not covered by historical rules
   - Correction: Discovers misjudgments or boundaries that are too wide/narrow in historical rules
   - Extension: Discovers that historical rules can be generalized to new categories/scenarios

2. **Essence Extraction**: Go beyond surface differences of specific cases to induce general task annotation logic (but don't over-speculate; the extracted logic should serve the task itself)

3. **Conflict Resolution**: If conflicts exist, analyze whether it's a "special handling of boundary cases" or a "fundamental flaw in historical rules", and provide a resolution plan

4. **Counterexample Testing**: Consider what scenarios would produce incorrect judgments if this rule were over-generalized, and tighten the applicable boundary accordingly

### Output Requirements
Note: Please analyze carefully based on the samples themselves. Since some samples may have deviations, please avoid using overly absolute judgment logic.
Please avoid academic or abstract vocabulary (such as "mechanism", "paradigm", "dimension", "attribute", etc.), and use plain language to induce general core annotation logic.

Please output your complete reasoning process first, then output the final JSON result:

Reasoning process: [Your reasoning process]

Final JSON format:
```json
{{
  "rule_intent": "Fused rule intent (reflecting core reasons and generalization principles, but don't deviate from the annotation task itself)",
  "decision_boundary": "Fused decision boundary (clearly define annotation logic)",
  "applicable_boundary": "Fused applicable boundary (define clear scope and exceptions based on sample types, don't over-associate)"
}}
```
""",

    # ============================================================================
    # Sample Context Prompts
    # ============================================================================

    "dimension_sample_prompt_header": """You are a professional annotation expert. Please annotate the following Query according to the rules below.

### Annotation Rules:
{base_prompt}
""",

    "dimension_sample_prompt_dimensions_section": """
### Key Discriminative Dimensions (Summarized from Historical Errors)
The following dimensions are associated with the provided reference samples. Please annotate by combining dimension understanding with sample demonstrations:
""",

    "dimension_sample_prompt_dimension_item": """
[Dimension {idx}: {dim_name}]
- Rule intent: {rule_intent}
- Decision boundary: {decision_boundary}
- Applicable scenario: {applicable_boundary}
""",

    "dimension_sample_prompt_context_section": """
### Reference Samples (Associated with the Above Dimensions)
{contrastive_context}

### Usage Guidance
1. First understand the rule intent and decision boundary of key discriminative dimensions
2. Refer to correct samples to understand how to judge, refer to incorrect samples to learn common pitfalls
3. Combine with the rule document to make an independent judgment on the current Query
""",

    "dimension_sample_prompt_footer": """
### Query to Annotate:
{query}

### Please output in the following format:
Reasoning process: [Step-by-step analysis of Query features, relevant dimensions, and comparison with reference samples]
Annotation result: {output_format}
""",

    "contrastive_sample_correct": """[Reference Sample {idx} (Correct Demonstration)]{dim_explanation}{quality_tag}{sim_text}
{image_placeholder}Query: {query}
Annotation result: {annotation}
Correct answer: {ground_truth}
""",

    "contrastive_sample_incorrect": """[Reference Sample {idx} (Incorrect Demonstration)]{dim_explanation}{quality_tag}{sim_text}
{image_placeholder}Query: {query}
Incorrect annotation: {annotation}
Correct answer: {ground_truth}
Error reason: {error_reason}
""",

    "contrastive_analysis_guidance": """[Contrastive Analysis Guidance]
Before making your judgment, carefully compare:
1. Core similarities between the current query and the {num_correct} correct sample(s)
2. Key differences between the current query and the {num_incorrect} incorrect sample(s)
3. How to avoid the mistakes made in the incorrect samples
4. Look for consistent patterns across multiple samples to improve judgment confidence
""",

    "contrastive_correct_only_guidance": "Please refer to the above {num_correct} correct demonstration(s) for annotation.\n",

    "contrastive_incorrect_only_guidance": "Please refer to the above {num_incorrect} incorrect demonstration(s) and avoid similar mistakes.\n",

    # ============================================================================
    # Optimization Prompts
    # ============================================================================

    "verifier_optimization_prompt": """You are a verification prompt optimization expert. Your task is to analyze current error samples, identify the root causes in the verification prompt that lead to wrong selections, and perform targeted optimization.

## Current Verification Prompt (to optimize)
{base_rules}

## Current Verification Error Samples (core optimization basis)
Please carefully analyze the following error samples, understand in which scenarios current verification prompt leads to wrong annotator selections:

{failure_samples}

## Error Reason Summary
{failure_reasons}

{reference_section}

## Please strictly output in the following format:
Analysis process: [First analyze the root cause of each error sample one by one, then combine with reference information to propose verification prompt optimization plan]
Optimized verification prompt: [Fine-tuned complete verification prompt, can be directly used by the verification agent]
""",

    "verifier_optimization_reference_section": """
## Auxiliary Reference Information
The following contrastive samples can help understand error patterns, but optimization should be primarily based on the error samples above:

#### Contrastive Sample Reference (Different judgments for similar Queries)
{contrastive_text}""",

    "annotator_optimization_prompt": """You are an annotation Prompt optimization expert. Your task is to analyze current error samples, identify the root causes in the Prompt that lead to annotation errors, and perform targeted optimization.

### Existing Annotation Prompt (to optimize):
{base_prompt}

{error_samples_section}

### Error Reason Summary:
{failure_reasons}

{reference_section}

### Please strictly output in the following format:
Analysis process: [First analyze the root cause of each error sample one by one, then combine with reference information to propose Prompt optimization suggestions]
Optimized Prompt: [Fine-tuned complete Prompt, can be directly used for annotation]
""",

    "annotator_optimization_error_samples_section": """### Current Annotation Error Samples (Core Optimization Basis)
Please carefully analyze the following error samples, understand in which scenarios the current Prompt leads to annotation errors:

{failure_samples}""",

    "annotator_optimization_reference_header": """
### Auxiliary Reference Information
The following dimensions and contrastive samples can help you understand error patterns, but optimization should be primarily based on the error samples above:
""",

    "annotator_optimization_contrastive_section": """
#### Contrastive Sample Reference (Different judgments for similar Queries)
{contrastive_text}""",

    "annotator_optimization_dimensions_section": """
#### Learned Discriminative Dimensions Reference
The following dimensions are general discriminative logic induced from historical contrastive samples, for reference only:
{dimension_text}""",

    "dimension_item_optimization": """[Dimension: {dim_name}] (Evidence count: {evidence_count})
- Rule intent: {rule_intent}
- Decision boundary: {decision_boundary}
- Applicable boundary: {applicable_boundary}
""",
    "reference_optimization_principles": """## Reference Optimization Principles
1. **Error samples as core**: Carefully analyze each error sample, understand why current prompt leads to these errors, optimization should directly solve these problems.
2. **Reference dimensions and contrastive samples**: Dimensions and contrastive samples provide summaries of historical error patterns, can serve as optimization reference, but should not overshadow the main focus.
3. **Fully integrate related suggestions, don't pile up at the end**: Avoid appending "additional notes", "additional clarifications", "supplementary rules" patch paragraphs at the end of the prompt.
4. **Prioritize reusing existing content**: Most optimization should be achieved by extending or rewriting existing prompt content when referencing dimensions and error samples.
5. **Eliminate contradictions**: When modifying or adding content based on dimensions and error samples, must check and synchronously correct other parts that may conflict.
""",


    # ============================================================================
    # SPO [Self-Supervised Prompt Optimization](https://aclanthology.org/2025.findings-emnlp.479/) templates
    # ============================================================================

    "spo_annotator_error_sample": """Sample {sid}:
Query to annotate: {query}{image_placeholder}
This annotator's output: {agent_output}
Expected correct answer: {ground_truth}""",

    "spo_verifier_error_sample": """Sample {sid}:
Query to annotate: {query}{image_placeholder}
Each annotator's output:
{annotations_text}Vote result: {vote_distribution}
Verifier selection: {verifier_selection}
Expected correct answer: {ground_truth}""",

    "spo_golden_text_item": "Sample {sid}: {ground_truth}",

    "spo_optimize_annotator_prompt": """You are building a prompt to address annotation requirements. Based on the given prompt, please reconstruct and optimize it.
You can add, modify, or delete prompt content. Please include a single modification in XML tags in your reply.
This is a prompt that performed excellently in a previous iteration. You must make further optimizations and improvements based on this prompt.
The modified prompt must differ from the provided example.

## Current Annotation Prompt:
```
{current_prompt}
```

## The execution result of this prompt (some error cases):
```
{error_text}
```

## The best answer we expect (for reference):
```
{golden_text}
```

Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyze what drawbacks exist in the results produced by the current prompt and how to improve them.</analyse>
<modification>Summarize the key points for improvement in one sentence</modification>
<prompt>Provide the complete optimized prompt here. It must be a full prompt that can be directly used for annotation, not a summary or fragment.</prompt>
""",

    "spo_optimize_verifier_prompt": """You are building a verification prompt to help a verification agent analyze multiple annotators' reasoning and select the best result.
Based on the given verification prompt, please reconstruct and optimize it.
You can add, modify, or delete prompt content. Please include a single modification in XML tags in your reply.
This is a verification prompt that performed excellently in a previous iteration. You must make further optimizations and improvements.
The modified verification prompt must differ from the provided example.

## Current Verification Prompt:
```
{verification_prompt_template}
```

## The execution result with this verification prompt (error cases where the verifier selected the wrong annotator, even though a correct annotator existed):
```
{error_text}
```

## The best answer we expect (for reference):
```
{golden_text}
```

Provide your analysis, optimization points, and the complete optimized verification prompt using the following XML format:
<analyse>Analyze what drawbacks exist in the current verification prompt that led to wrong selections and how to improve them.</analyse>
<modification>Summarize the key points for improvement in one sentence</modification>
<prompt>Provide the complete optimized verification prompt here. It must be a complete prompt that can be directly used by the verification agent, not a summary or fragment.</prompt>
"""
}

# For backward compatibility
PROMPTS = PROMPTS_EN


# ============================================================================
# Output Format Templates (by task type)
# ============================================================================

OUTPUT_FORMATS = {
    "disease_privacy_assessment": """```json
{"隐私合规评估": "有/无问题"}```""",
    "medical_query_evaluation": """```json
{"隐私合规评估": "有/无问题"}```""",
    "billing_scenario_classification": """```json
{"账单类目": "标注结果"}```""",
    "social_text_classification": """```json
{"label": "分类结果"}```""",
    "entity_extraction": """```json
{{
"entities": [
        {
            "text": "实体文本",
            "category": "类别名"
        }
    ]
}}```""",
    "image_classification": """```json\n{"一级品类": "[分类结果]", "二级品类": "[分类结果]", "话术分析": "[导向类型]"}```""",
    "image_text_relevance": """```json\n{"label": "[相关性打分]"}```""",
    "image_attribution_recognition": """```json\n{"label": "[可识别属性数量(至多为2)"}]}```""",
}


# ============================================================================
# Prompt Builder Functions
# ============================================================================

def build_dimension_sample_prompt(
    base_prompt: str,
    query: str,
    contrastive_context: str,
    sample_dimensions: list,
    output_format: str,
    prompts_dict: dict = None
) -> str:
    """Build a prompt with dimension and sample context for annotation.

    Args:
        base_prompt: The base annotation rules/prompt
        query: The query to annotate
        contrastive_context: Formatted contrastive sample context
        sample_dimensions: List of dimension dictionaries
        output_format: The expected output format
        prompts_dict: The prompts dictionary to use (defaults to PROMPTS_EN)

    Returns:
        The complete prompt string
    """
    if prompts_dict is None:
        prompts_dict = PROMPTS_EN

    parts = [prompts_dict["dimension_sample_prompt_header"].format(base_prompt=base_prompt)]

    if sample_dimensions:
        parts.append(prompts_dict["dimension_sample_prompt_dimensions_section"])
        for i, dim in enumerate(sample_dimensions[:5], 1):
            parts.append(prompts_dict["dimension_sample_prompt_dimension_item"].format(
                idx=i,
                dim_name=dim.get('name', ''),
                rule_intent=dim.get('rule_intent', 'None yet'),
                decision_boundary=dim.get('decision_boundary', 'None yet'),
                applicable_boundary=dim.get('applicable_boundary', 'Related scenarios')
            ))

    if contrastive_context:
        parts.append(prompts_dict["dimension_sample_prompt_context_section"].format(
            contrastive_context=contrastive_context
        ))

    parts.append(prompts_dict["dimension_sample_prompt_footer"].format(
        query=query,
        output_format=output_format
    ))

    return "\n".join(parts)


def build_contrastive_context(
    correct_samples: list,
    incorrect_samples: list,
    similarity_scores: dict = None,
    relevant_dimensions: list = None,
    include_images: bool = False,
    return_image_urls: bool = False,
    prompts_dict: dict = None
) -> str:
    """Build contrastive context string from correct and incorrect samples.

    Args:
        correct_samples: List of correct sample dictionaries
        incorrect_samples: List of incorrect sample dictionaries
        similarity_scores: Dictionary with similarity scores
        relevant_dimensions: List of relevant dimension dictionaries
        include_images: Whether to include image placeholders
        return_image_urls: Whether to return collected image URLs
        prompts_dict: The prompts dictionary to use (defaults to PROMPTS_EN)

    Returns:
        The formatted contrastive context string, or (context_str, image_urls) if return_image_urls=True
    """
    if prompts_dict is None:
        prompts_dict = PROMPTS_EN

    if similarity_scores is None:
        similarity_scores = {}

    context_parts = []
    all_image_urls = []

    dim_explanation = ""
    if relevant_dimensions:
        dim_names = [d.get('name', '') for d in relevant_dimensions[:3] if d.get('name')]
        if dim_names:
            dim_explanation = f" (Related dimensions: {', '.join(dim_names)})"

    correct_sims = similarity_scores.get('correct_all', [])
    for i, sample in enumerate(correct_samples):
        sim_score = correct_sims[i] if i < len(correct_sims) else 0.0
        image_placeholder = ""
        if include_images:
            sample_images = sample.get("image_urls", [])
            if sample_images:
                image_placeholder = "\n".join(["<image>" for _ in sample_images]) + "\n"
                all_image_urls.extend(sample_images)

        context_parts.append(prompts_dict["contrastive_sample_correct"].format(
            idx=i + 1,
            dim_explanation=dim_explanation,
            quality_tag="",
            sim_text=f" (similarity: {sim_score:.3f})",
            image_placeholder=image_placeholder,
            query=sample.get("query", ""),
            annotation=sample.get("annotation", ""),
            ground_truth=sample.get("ground_truth", "")
        ))

    incorrect_sims = similarity_scores.get('incorrect_all', [])
    for i, sample in enumerate(incorrect_samples):
        sim_score = incorrect_sims[i] if i < len(incorrect_sims) else 0.0
        image_placeholder = ""
        if include_images:
            sample_images = sample.get("image_urls", [])
            if sample_images:
                image_placeholder = "\n".join(["<image>" for _ in sample_images]) + "\n"
                all_image_urls.extend(sample_images)

        context_parts.append(prompts_dict["contrastive_sample_incorrect"].format(
            idx=i + 1,
            dim_explanation=dim_explanation,
            quality_tag="",
            sim_text=f" (similarity: {sim_score:.3f})",
            image_placeholder=image_placeholder,
            query=sample.get("query", ""),
            annotation=sample.get("annotation", sample.get("error_output", "")),
            ground_truth=sample.get("ground_truth", ""),
            error_reason=sample.get("error_reason", "Unknown")
        ))

    if correct_samples and incorrect_samples:
        context_parts.append(prompts_dict["contrastive_analysis_guidance"].format(
            num_correct=len(correct_samples),
            num_incorrect=len(incorrect_samples)
        ))
    elif correct_samples:
        context_parts.append(prompts_dict["contrastive_correct_only_guidance"].format(
            num_correct=len(correct_samples)
        ))
    elif incorrect_samples:
        context_parts.append(prompts_dict["contrastive_incorrect_only_guidance"].format(
            num_incorrect=len(incorrect_samples)
        ))

    context_str = "\n\n".join(context_parts)
    if return_image_urls:
        return context_str, all_image_urls
    return context_str


def build_verification_prompt(
    verification_prompt_template: str,
    query_str: str,
    annotations_text: str,
    vote_text: str,
    sample_section: str = "",
    dimension_section: str = "",
    prompts_dict: dict = None
) -> str:
    """Build the verification prompt.

    Args:
        verification_prompt_template: The base verification prompt template
        query_str: The query string
        annotations_text: Formatted annotations from all annotators
        vote_text: Vote information text
        sample_section: Optional sample reference section
        dimension_section: Optional dimension section
        prompts_dict: The prompts dictionary to use (defaults to PROMPTS_EN)

    Returns:
        The complete verification prompt string
    """
    if prompts_dict is None:
        prompts_dict = PROMPTS_EN

    return prompts_dict["verification_prompt_format"].format(
        verification_prompt_template=verification_prompt_template,
        query_str=query_str,
        dimension_section=dimension_section,
        annotations_text=annotations_text,
        vote_text=vote_text,
        sample_section=sample_section
    )


def build_dimension_aware_framework(dimensions: list) -> str:
    """Build the dimension-aware analysis section to append to a base prompt.

    Each dimension dict should have keys: name, meta_rule, rule_intent,
    applicable_boundary, decision_boundary, evidence_count.
    """
    if not dimensions:
        return ""

    parts = [
        "\n\n## [Decision-Preceding Analysis - Key Learned Discriminative Dimensions]",
        "Based on historical error analysis, the following discriminative "
        "dimensions are crucial for correct annotation. Please evaluate them "
        "carefully before making final judgments:\n",
    ]

    for i, dim in enumerate(dimensions, 1):
        dim_name = dim.get("name", "")
        meta_rule = dim.get("meta_rule", "")
        rule_intent = dim.get("rule_intent", "")
        decision_boundary = dim.get("decision_boundary", "")
        applicable_boundary = dim.get("applicable_boundary", "")
        evidence_count = dim.get("evidence_count", 0)

        parts.append(f"**Dimension {i}: {dim_name}**")
        if meta_rule and meta_rule != dim_name:
            parts.append(f"- Meta-rule: {meta_rule}")
        if rule_intent:
            parts.append(f"- Rule intent: {rule_intent}")
        if applicable_boundary:
            parts.append(f"- Applicable boundary: {applicable_boundary}")
        if decision_boundary:
            parts.append(f"- Decision boundary: {decision_boundary}")
        parts.append(f"- Confidence: Based on analysis of {evidence_count} samples\n")

    parts.append(
        "Please confirm that all the above dimensions have been fully "
        "considered before outputting the final answer, especially the "
        "rule intent and decision boundary.\n"
    )

    return "\n".join(parts)
